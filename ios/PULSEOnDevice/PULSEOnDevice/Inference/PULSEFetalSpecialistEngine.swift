import CoreGraphics
import CoreML
import Foundation
import UIKit

struct PULSEFetalSpecialistManifest: Decodable {
    let name: String
    let mobileFetalClip: MobileFetalClipConfig?
    let fetalnet: FetalNetConfig?

    enum CodingKeys: String, CodingKey {
        case name
        case mobileFetalClip = "mobile_fetal_clip"
        case fetalnet
    }

    struct MobileFetalClipConfig: Decodable {
        let modelName: String
        let inputSize: Int
        let embeddingDim: Int
        let promptSets: [String: PromptSet]

        enum CodingKeys: String, CodingKey {
            case modelName = "model_name"
            case inputSize = "input_size"
            case embeddingDim = "embedding_dim"
            case promptSets = "prompt_sets"
        }
    }

    struct PromptSet: Decodable {
        let labels: [String]
        let displayMap: [String: String]
        let embeddings: [[Double]]

        enum CodingKeys: String, CodingKey {
            case labels
            case displayMap = "display_map"
            case embeddings
        }
    }

    struct FetalNetConfig: Decodable {
        let modelName: String
        let inputSize: Int
        let classLabels: [String]

        enum CodingKeys: String, CodingKey {
            case modelName = "model_name"
            case inputSize = "input_size"
            case classLabels = "class_labels"
        }
    }

    static func loadFromBundle() throws -> PULSEFetalSpecialistManifest {
        guard let url = Bundle.main.findResource(named: "fetal_specialists_manifest", extension: "json", preferredSubdirectories: ["Models", "Resources/Models"]) else {
            throw PULSEOnDeviceError.missingManifest
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PULSEFetalSpecialistManifest.self, from: data)
    }
}

actor PULSEFetalSpecialistRegistry {
    static let shared = PULSEFetalSpecialistRegistry()

    private let manifest: PULSEFetalSpecialistManifest?
    private var cache: [String: MLModel] = [:]

    init() {
        manifest = try? PULSEFetalSpecialistManifest.loadFromBundle()
    }

    func currentManifest() -> PULSEFetalSpecialistManifest? {
        manifest
    }

    func loadModel(named modelName: String) throws -> MLModel {
        if let cached = cache[modelName] {
            return cached
        }
        guard let url =
            Bundle.main.findResource(named: modelName, extension: "mlmodelc", preferredSubdirectories: ["Models", "Resources/Models"]) ??
            Bundle.main.findResource(named: modelName, extension: "mlpackage", preferredSubdirectories: ["Models", "Resources/Models"])
        else {
            throw PULSEOnDeviceError.missingModel(modelName)
        }
        let model = try MLModel(contentsOf: url)
        cache[modelName] = model
        return model
    }
}

struct PULSEFetalSpecialistRun {
    let findings: [PULSEFinding]
    let detectedLabel: String?
}

private struct PULSEFetalSubviewResult {
    let viewKey: String?
    let displayLabel: String?
    let confidence: Double?
    let details: [String]
}

private struct PULSEFetalMeasurementResult {
    let classificationLabel: String?
    let classificationConfidence: Double?
    let measurementFinding: PULSEFinding?
}

private struct PULSEFetalNetInference {
    let predictedClass: String
    let confidence: Double
    let probabilities: [Double]
    let maskProbabilities: [Double]
    let maskWidth: Int
    let maskHeight: Int
}

extension PULSEOnDevicePipeline {
    func runFetalSpecialists(primaryImage: UIImage) async throws -> PULSEFetalSpecialistRun {
        let registry = PULSEFetalSpecialistRegistry.shared
        guard let manifest = await registry.currentManifest() else {
            return PULSEFetalSpecialistRun(findings: [], detectedLabel: nil)
        }

        var findings: [PULSEFinding] = []
        var detectedLabel: String?
        var fetalSubview: PULSEFetalSubviewResult?
        var fetalMeasurement: PULSEFetalMeasurementResult?

        if let mobileConfig = manifest.mobileFetalClip {
            fetalSubview = try await runMobileFetalClip(primaryImage: primaryImage, config: mobileConfig, registry: registry)
        }

        if let fetalNetConfig = manifest.fetalnet {
            fetalMeasurement = try await runFetalNet(
                primaryImage: primaryImage,
                expectedViewKey: fetalSubview?.viewKey,
                config: fetalNetConfig,
                registry: registry
            )
        }

        if let fetalSubview {
            findings.append(
                PULSEFinding(
                    taskID: "fetal/subview",
                    title: "Subview",
                    summary: "Detected subview: \(fetalSubview.displayLabel?.pulseDisplayText ?? "Unknown").",
                    label: fetalSubview.displayLabel,
                    confidence: fetalSubview.confidence,
                    details: fetalSubview.details
                )
            )
            detectedLabel = fetalSubview.displayLabel
        } else if let fallbackLabel = fetalMeasurement?.classificationLabel {
            findings.append(
                PULSEFinding(
                    taskID: "fetal/subview",
                    title: "Subview",
                    summary: "Detected subview: \(fallbackLabel.pulseDisplayText).",
                    label: fallbackLabel,
                    confidence: fetalMeasurement?.classificationConfidence,
                    details: []
                )
            )
            detectedLabel = fallbackLabel
        }

        if let measurementFinding = fetalMeasurement?.measurementFinding {
            findings.append(measurementFinding)
        }

        return PULSEFetalSpecialistRun(findings: findings, detectedLabel: detectedLabel)
    }

    private func runMobileFetalClip(
        primaryImage: UIImage,
        config: PULSEFetalSpecialistManifest.MobileFetalClipConfig,
        registry: PULSEFetalSpecialistRegistry
    ) async throws -> PULSEFetalSubviewResult {
        let model = try await registry.loadModel(named: config.modelName)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [
                "image": MLFeatureValue(pixelBuffer: try ImageBufferFactory.pixelBuffer(from: primaryImage, size: config.inputSize))
            ]
        )
        let output = try await model.prediction(from: provider)
        guard let embeddingValue = output.featureValue(for: "image_embedding") else {
            throw PULSEOnDeviceError.invalidPrediction("MobileFetalCLIP output `image_embedding` is missing.")
        }
        let embedding = try fetalFlattenedArray(from: embeddingValue)
        let planePromptSet = config.promptSets["five_plane"]
        guard let planePromptSet else {
            throw PULSEOnDeviceError.invalidPrediction("MobileFetalCLIP five-plane prompt set is missing from the specialist manifest.")
        }

        let planeResult = classifyPromptBank(embedding: embedding, promptSet: planePromptSet)
        var selectedLabel = planeResult.displayLabel.pulseTrimmed(for: "fetal")
        var selectedConfidence = planeResult.confidence
        var details: [String] = []

        if planeResult.labelKey == "brain", let brainPromptSet = config.promptSets["brain_subplane"] {
            let brainResult = classifyPromptBank(embedding: embedding, promptSet: brainPromptSet)
            selectedLabel = brainResult.displayLabel.pulseTrimmed(for: "fetal")
            selectedConfidence = brainResult.confidence
            details.append("brain_subview: \(brainResult.displayLabel.pulseDisplayText)")
        }

        return PULSEFetalSubviewResult(
            viewKey: planeResult.labelKey,
            displayLabel: selectedLabel,
            confidence: selectedConfidence,
            details: details
        )
    }

    private func runFetalNet(
        primaryImage: UIImage,
        expectedViewKey: String?,
        config: PULSEFetalSpecialistManifest.FetalNetConfig,
        registry: PULSEFetalSpecialistRegistry
    ) async throws -> PULSEFetalMeasurementResult {
        let inference = try await fetalNetInference(primaryImage: primaryImage, config: config, registry: registry)
        let labelMap = [
            "head": "brain",
            "abdomen": "abdomen",
            "femur": "femur",
            "background": "background"
        ]

        let expectedClass = expectedClassForViewKey(expectedViewKey)
        guard let expectedClass else {
            return PULSEFetalMeasurementResult(
                classificationLabel: inference.predictedClass == "background" ? nil : (labelMap[inference.predictedClass] ?? inference.predictedClass),
                classificationConfidence: inference.confidence,
                measurementFinding: nil
            )
        }
        let expectedConfidence = expectedConfidenceForViewKey(expectedClass: expectedClass, probabilities: inference.probabilities, classLabels: config.classLabels)
        if inference.predictedClass != expectedClass && expectedConfidence < 0.35 {
            return PULSEFetalMeasurementResult(
                classificationLabel: labelMap[inference.predictedClass] ?? inference.predictedClass,
                classificationConfidence: inference.confidence,
                measurementFinding: nil
            )
        }

        let restoredMask = try restoreMask(
            probabilities: inference.maskProbabilities,
            maskWidth: inference.maskWidth,
            maskHeight: inference.maskHeight,
            originalSize: primaryImage.size
        )
        let binaryMask = restoredMask.map { $0 > 0.6 ? UInt8(1) : UInt8(0) }
        guard binaryMask.reduce(0, { $0 + Int($1) }) >= 20 else {
            return PULSEFetalMeasurementResult(
                classificationLabel: labelMap[inference.predictedClass] ?? inference.predictedClass,
                classificationConfidence: inference.confidence,
                measurementFinding: nil
            )
        }

        let measurement = try fetalMeasurement(
            binaryMask: binaryMask,
            width: Int(primaryImage.size.width.rounded()),
            height: Int(primaryImage.size.height.rounded()),
            expectedClass: expectedClass
        )
        let artifacts = fetalNetArtifacts(
            primaryImage: primaryImage,
            binaryMask: binaryMask,
            width: Int(primaryImage.size.width.rounded()),
            height: Int(primaryImage.size.height.rounded()),
            measurement: measurement,
            expectedClass: expectedClass
        )
        let finding = PULSEFinding(
            taskID: fetalNetTaskID(for: expectedClass),
            title: "Biometry",
            summary: fetalNetSummary(measurement: measurement, expectedClass: expectedClass),
            label: measurement.measurementName,
            confidence: expectedConfidence,
            details: fetalNetMeasurementDetails(measurement: measurement, expectedClass: expectedClass),
            artifacts: artifacts
        )

        return PULSEFetalMeasurementResult(
            classificationLabel: labelMap[inference.predictedClass] ?? inference.predictedClass,
            classificationConfidence: inference.confidence,
            measurementFinding: finding
        )
    }

    private func fetalNetInference(
        primaryImage: UIImage,
        config: PULSEFetalSpecialistManifest.FetalNetConfig,
        registry: PULSEFetalSpecialistRegistry
    ) async throws -> PULSEFetalNetInference {
        let model = try await registry.loadModel(named: config.modelName)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [
                "image": MLFeatureValue(pixelBuffer: try ImageBufferFactory.pixelBuffer(from: primaryImage, size: config.inputSize))
            ]
        )
        let output = try await model.prediction(from: provider)
        guard
            let logitsValue = output.featureValue(for: "view_logits"),
            let maskValue = output.featureValue(for: "mask_logits")
        else {
            throw PULSEOnDeviceError.invalidPrediction("FetalNet outputs are missing from the Core ML model.")
        }

        let logits = try fetalFlattenedArray(from: logitsValue)
        let probabilities = fetalSoftmax(logits)
        let bestIndex = probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        let predictedClass = config.classLabels.indices.contains(bestIndex) ? config.classLabels[bestIndex] : "background"
        let maskTensor = try fetalTensor(from: maskValue)
        let spatial = try fetalSpatialDimensions(shape: maskTensor.shape)
        return PULSEFetalNetInference(
            predictedClass: predictedClass,
            confidence: probabilities.indices.contains(bestIndex) ? probabilities[bestIndex] : 0.0,
            probabilities: probabilities,
            maskProbabilities: maskTensor.spatialSliceValues,
            maskWidth: spatial.width,
            maskHeight: spatial.height
        )
    }
}

private extension PULSEOnDevicePipeline {
    struct PromptClassificationResult {
        let labelKey: String
        let displayLabel: String
        let confidence: Double
    }

    struct FetalMeasurement {
        let centerX: Double
        let centerY: Double
        let majorAxisPixels: Double
        let minorAxisPixels: Double
        let primaryAxis: (x: Double, y: Double)
        let secondaryAxis: (x: Double, y: Double)
        let minPrimary: Double
        let maxPrimary: Double
        let value: Double?
        let measurementName: String?
        let hcValue: Double?
        let bpdValue: Double?
    }

    func classifyPromptBank(
        embedding: [Double],
        promptSet: PULSEFetalSpecialistManifest.PromptSet
    ) -> PromptClassificationResult {
        let normalizedEmbedding = fetalNormalize(embedding)
        let logits = promptSet.embeddings.map { row in
            zip(normalizedEmbedding, row).reduce(0.0) { partial, pair in
                partial + (pair.0 * pair.1)
            } * 100.0
        }
        let probabilities = fetalSoftmax(logits)
        let bestIndex = probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        let labelKey = promptSet.labels.indices.contains(bestIndex) ? promptSet.labels[bestIndex] : "unknown"
        let displayLabel = promptSet.displayMap[labelKey] ?? labelKey
        return PromptClassificationResult(
            labelKey: labelKey,
            displayLabel: displayLabel,
            confidence: probabilities.indices.contains(bestIndex) ? probabilities[bestIndex] : 0.0
        )
    }

    func expectedClassForViewKey(_ viewKey: String?) -> String? {
        switch viewKey {
        case "brain":
            return "head"
        case "abdomen":
            return "abdomen"
        case "femur":
            return "femur"
        default:
            return nil
        }
    }

    func expectedConfidenceForViewKey(expectedClass: String?, probabilities: [Double], classLabels: [String]) -> Double {
        guard let expectedClass, let index = classLabels.firstIndex(of: expectedClass), probabilities.indices.contains(index) else {
            return 0.0
        }
        return probabilities[index]
    }

    func fetalNetTaskID(for expectedClass: String) -> String {
        switch expectedClass {
        case "head":
            return "fetalnet/head_biometry"
        case "abdomen":
            return "fetalnet/abdominal_circumference"
        case "femur":
            return "fetalnet/femur_length"
        default:
            return "fetalnet/measurement"
        }
    }

    func fetalNetTitle(for expectedClass: String) -> String {
        switch expectedClass {
        case "head":
            return "Biometry"
        case "abdomen":
            return "Biometry"
        case "femur":
            return "Biometry"
        default:
            return "Biometry"
        }
    }

    func fetalNetSummary(measurement: FetalMeasurement, expectedClass: String) -> String {
        switch expectedClass {
        case "head":
            return "Head biometry generated for the fetal brain view."
        case "abdomen", "femur":
            return "Measurement generated for the detected fetal subview."
        default:
            return "Measurement generated."
        }
    }

    func fetalNetMeasurementDetails(measurement: FetalMeasurement, expectedClass: String) -> [String] {
        switch expectedClass {
        case "head":
            return [
                "HC: \(fetalFormat(measurement.hcValue ?? 0)) pixels",
                "BPD: \(fetalFormat(measurement.bpdValue ?? 0)) pixels",
                "major_axis: \(fetalFormat(measurement.majorAxisPixels)) pixels",
                "minor_axis: \(fetalFormat(measurement.minorAxisPixels)) pixels",
            ]
        case "abdomen", "femur":
            return [
                "\(measurement.measurementName ?? "value"): \(fetalFormat(measurement.value ?? 0)) pixels",
                "major_axis: \(fetalFormat(measurement.majorAxisPixels)) pixels",
                "minor_axis: \(fetalFormat(measurement.minorAxisPixels)) pixels",
            ]
        default:
            return []
        }
    }

    func restoreMask(
        probabilities: [Double],
        maskWidth: Int,
        maskHeight: Int,
        originalSize: CGSize
    ) throws -> [Double] {
        guard let maskImage = grayscaleImage(values: probabilities, width: maskWidth, height: maskHeight) else {
            throw PULSEOnDeviceError.invalidPrediction("FetalNet mask could not be reconstructed.")
        }

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        let originalWidth = max(Int(originalSize.width.rounded()), 1)
        let originalHeight = max(Int(originalSize.height.rounded()), 1)
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: originalWidth, height: originalHeight), format: format)
        let restoredImage = renderer.image { _ in
            UIImage(cgImage: maskImage).draw(in: CGRect(x: 0, y: 0, width: originalWidth, height: originalHeight))
        }
        guard let restoredCGImage = restoredImage.cgImage else {
            throw PULSEOnDeviceError.invalidPrediction("Fetal mask restoration failed.")
        }
        return try grayscalePixels(from: restoredCGImage)
    }

    func fetalMeasurement(
        binaryMask: [UInt8],
        width: Int,
        height: Int,
        expectedClass: String
    ) throws -> FetalMeasurement {
        var points: [(x: Double, y: Double)] = []
        points.reserveCapacity(binaryMask.count / 6)
        for y in 0..<height {
            for x in 0..<width {
                if binaryMask[(y * width) + x] > 0 {
                    points.append((Double(x), Double(y)))
                }
            }
        }
        guard points.count >= 3 else {
            throw PULSEOnDeviceError.invalidPrediction("FetalNet measurement requires at least three foreground pixels.")
        }

        let centerX = points.reduce(0.0) { $0 + $1.x } / Double(points.count)
        let centerY = points.reduce(0.0) { $0 + $1.y } / Double(points.count)

        var covXX = 0.0
        var covXY = 0.0
        var covYY = 0.0
        for point in points {
            let dx = point.x - centerX
            let dy = point.y - centerY
            covXX += dx * dx
            covXY += dx * dy
            covYY += dy * dy
        }
        let denom = max(Double(points.count - 1), 1.0)
        covXX /= denom
        covXY /= denom
        covYY /= denom

        let (primaryAxis, secondaryAxis) = eigenvectors2x2(a: covXX, b: covXY, d: covYY)

        var minPrimary = Double.greatestFiniteMagnitude
        var maxPrimary = -Double.greatestFiniteMagnitude
        var minSecondary = Double.greatestFiniteMagnitude
        var maxSecondary = -Double.greatestFiniteMagnitude
        for point in points {
            let dx = point.x - centerX
            let dy = point.y - centerY
            let primary = (dx * primaryAxis.x) + (dy * primaryAxis.y)
            let secondary = (dx * secondaryAxis.x) + (dy * secondaryAxis.y)
            minPrimary = min(minPrimary, primary)
            maxPrimary = max(maxPrimary, primary)
            minSecondary = min(minSecondary, secondary)
            maxSecondary = max(maxSecondary, secondary)
        }

        let majorAxis = maxPrimary - minPrimary + 1.0
        let minorAxis = maxSecondary - minSecondary + 1.0
        let circumference = fetalEllipseCircumference(majorAxis: majorAxis, minorAxis: minorAxis)

        switch expectedClass {
        case "head":
            return FetalMeasurement(
                centerX: centerX,
                centerY: centerY,
                majorAxisPixels: majorAxis,
                minorAxisPixels: minorAxis,
                primaryAxis: primaryAxis,
                secondaryAxis: secondaryAxis,
                minPrimary: minPrimary,
                maxPrimary: maxPrimary,
                value: nil,
                measurementName: nil,
                hcValue: circumference,
                bpdValue: minorAxis
            )
        case "abdomen":
            return FetalMeasurement(
                centerX: centerX,
                centerY: centerY,
                majorAxisPixels: majorAxis,
                minorAxisPixels: minorAxis,
                primaryAxis: primaryAxis,
                secondaryAxis: secondaryAxis,
                minPrimary: minPrimary,
                maxPrimary: maxPrimary,
                value: circumference,
                measurementName: "AC",
                hcValue: nil,
                bpdValue: nil
            )
        default:
            return FetalMeasurement(
                centerX: centerX,
                centerY: centerY,
                majorAxisPixels: majorAxis,
                minorAxisPixels: minorAxis,
                primaryAxis: primaryAxis,
                secondaryAxis: secondaryAxis,
                minPrimary: minPrimary,
                maxPrimary: maxPrimary,
                value: majorAxis,
                measurementName: "FL",
                hcValue: nil,
                bpdValue: nil
            )
        }
    }

    func fetalNetArtifacts(
        primaryImage: UIImage,
        binaryMask: [UInt8],
        width: Int,
        height: Int,
        measurement: FetalMeasurement,
        expectedClass: String
    ) -> [PULSEArtifactImage] {
        guard
            let maskImage = makeFetalMaskImage(binaryMask: binaryMask, width: width, height: height),
            let overlayImage = makeFetalOverlayImage(
                primaryImage: primaryImage,
                binaryMask: binaryMask,
                width: width,
                height: height,
                measurement: measurement,
                expectedClass: expectedClass
            ),
            let maskData = maskImage.pngData(),
            let overlayData = overlayImage.pngData()
        else {
            return []
        }

        return [
            PULSEArtifactImage(kind: "overlay", caption: "Measurement overlay", pngData: overlayData),
            PULSEArtifactImage(kind: "mask", caption: "Segmentation mask", pngData: maskData),
        ]
    }

    func makeFetalMaskImage(binaryMask: [UInt8], width: Int, height: Int) -> UIImage? {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for index in 0..<(width * height) {
            let offset = index * 4
            if binaryMask[index] > 0 {
                pixels[offset] = 18
                pixels[offset + 1] = 186
                pixels[offset + 2] = 150
                pixels[offset + 3] = 255
            } else {
                pixels[offset] = 9
                pixels[offset + 1] = 23
                pixels[offset + 2] = 38
                pixels[offset + 3] = 255
            }
        }
        return makeRGBAImage(pixels: pixels, width: width, height: height)
    }

    func makeFetalOverlayImage(
        primaryImage: UIImage,
        binaryMask: [UInt8],
        width: Int,
        height: Int,
        measurement: FetalMeasurement,
        expectedClass: String
    ) -> UIImage? {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        return renderer.image { _ in
            primaryImage.draw(in: CGRect(x: 0, y: 0, width: width, height: height))

            if let overlay = makeFetalTransparentMask(binaryMask: binaryMask, width: width, height: height) {
                overlay.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
            }

            let path = UIBezierPath()
            UIColor(red: 1.0, green: 0.84, blue: 0.12, alpha: 1.0).setStroke()
            path.lineWidth = 3.0

            if expectedClass == "femur" {
                let start = CGPoint(
                    x: measurement.centerX + (measurement.primaryAxis.x * measurement.minPrimary),
                    y: measurement.centerY + (measurement.primaryAxis.y * measurement.minPrimary)
                )
                let end = CGPoint(
                    x: measurement.centerX + (measurement.primaryAxis.x * measurement.maxPrimary),
                    y: measurement.centerY + (measurement.primaryAxis.y * measurement.maxPrimary)
                )
                path.move(to: start)
                path.addLine(to: end)
                path.stroke()
            } else {
                let pointCount = 96
                for index in 0..<pointCount {
                    let angle = (Double(index) / Double(pointCount)) * 2.0 * Double.pi
                    let primary = (measurement.majorAxisPixels / 2.0) * cos(angle)
                    let secondary = (measurement.minorAxisPixels / 2.0) * sin(angle)
                    let x = measurement.centerX + (measurement.primaryAxis.x * primary) + (measurement.secondaryAxis.x * secondary)
                    let y = measurement.centerY + (measurement.primaryAxis.y * primary) + (measurement.secondaryAxis.y * secondary)
                    let point = CGPoint(x: x, y: y)
                    if index == 0 {
                        path.move(to: point)
                    } else {
                        path.addLine(to: point)
                    }
                }
                path.close()
                path.stroke()

                if expectedClass == "head" {
                    let start = CGPoint(
                        x: measurement.centerX - (measurement.secondaryAxis.x * (measurement.minorAxisPixels / 2.0)),
                        y: measurement.centerY - (measurement.secondaryAxis.y * (measurement.minorAxisPixels / 2.0))
                    )
                    let end = CGPoint(
                        x: measurement.centerX + (measurement.secondaryAxis.x * (measurement.minorAxisPixels / 2.0)),
                        y: measurement.centerY + (measurement.secondaryAxis.y * (measurement.minorAxisPixels / 2.0))
                    )
                    let bpdPath = UIBezierPath()
                    UIColor.systemRed.setStroke()
                    bpdPath.lineWidth = 3.0
                    bpdPath.move(to: start)
                    bpdPath.addLine(to: end)
                    bpdPath.stroke()
                }
            }
        }
    }

    func makeFetalTransparentMask(binaryMask: [UInt8], width: Int, height: Int) -> UIImage? {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for index in 0..<(width * height) {
            guard binaryMask[index] > 0 else { continue }
            let offset = index * 4
            pixels[offset] = 0
            pixels[offset + 1] = 255
            pixels[offset + 2] = 0
            pixels[offset + 3] = 84
        }
        return makeRGBAImage(pixels: pixels, width: width, height: height)
    }

    func makeRGBAImage(pixels: [UInt8], width: Int, height: Int) -> UIImage? {
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData) else {
            return nil
        }
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }

    func grayscaleImage(values: [Double], width: Int, height: Int) -> CGImage? {
        let bytes = values.map { UInt8(max(0, min(255, Int(($0 * 255.0).rounded())))) }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else {
            return nil
        }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    func grayscalePixels(from cgImage: CGImage) throws -> [Double] {
        let width = cgImage.width
        let height = cgImage.height
        var bytes = [UInt8](repeating: 0, count: width * height)
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw PULSEOnDeviceError.invalidPrediction("Could not decode grayscale pixels.")
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return bytes.map { Double($0) / 255.0 }
    }

    func eigenvectors2x2(a: Double, b: Double, d: Double) -> ((x: Double, y: Double), (x: Double, y: Double)) {
        let trace = a + d
        let delta = sqrt(max((a - d) * (a - d) + (4.0 * b * b), 0.0))
        let lambda1 = (trace + delta) / 2.0

        let primary: (Double, Double)
        if abs(b) > 1e-8 {
            primary = normalizeVector(x: b, y: lambda1 - a)
        } else if a >= d {
            primary = (1.0, 0.0)
        } else {
            primary = (0.0, 1.0)
        }
        let secondary = (-primary.1, primary.0)
        return ((primary.0, primary.1), (secondary.0, secondary.1))
    }

    func normalizeVector(x: Double, y: Double) -> (Double, Double) {
        let norm = max(sqrt((x * x) + (y * y)), 1e-8)
        return (x / norm, y / norm)
    }

    func fetalEllipseCircumference(majorAxis: Double, minorAxis: Double) -> Double {
        let a = max(majorAxis, minorAxis) / 2.0
        let b = min(majorAxis, minorAxis) / 2.0
        let h = ((a - b) * (a - b)) / max(((a + b) * (a + b)), 1e-8)
        return Double.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + sqrt(max(4.0 - (3.0 * h), 1e-8))))
    }

    func fetalNormalize(_ values: [Double]) -> [Double] {
        let norm = max(sqrt(values.reduce(0.0) { $0 + ($1 * $1) }), 1e-8)
        return values.map { $0 / norm }
    }

    func fetalSoftmax(_ logits: [Double]) -> [Double] {
        let maximum = logits.max() ?? 0.0
        let exps = logits.map { Foundation.exp($0 - maximum) }
        let sum = exps.reduce(0.0, +)
        return exps.map { $0 / max(sum, 1e-12) }
    }

    func fetalFlattenedArray(from featureValue: MLFeatureValue) throws -> [Double] {
        try fetalTensor(from: featureValue).values
    }

    func fetalTensor(from featureValue: MLFeatureValue) throws -> (values: [Double], shape: [Int], spatialSliceValues: [Double]) {
        guard let multiArray = featureValue.multiArrayValue else {
            throw PULSEOnDeviceError.invalidPrediction("Expected MLMultiArray output.")
        }
        let shape = multiArray.shape.map(\.intValue)
        let values: [Double]
        switch multiArray.dataType {
        case .double:
            let pointer = multiArray.dataPointer.bindMemory(to: Double.self, capacity: multiArray.count)
            values = Array(UnsafeBufferPointer(start: pointer, count: multiArray.count))
        case .float32:
            let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)
            values = Array(UnsafeBufferPointer(start: pointer, count: multiArray.count)).map(Double.init)
        case .float16:
            let pointer = multiArray.dataPointer.bindMemory(to: UInt16.self, capacity: multiArray.count)
            values = Array(UnsafeBufferPointer(start: pointer, count: multiArray.count)).map { Double(Float16(bitPattern: $0)) }
        case .int32:
            let pointer = multiArray.dataPointer.bindMemory(to: Int32.self, capacity: multiArray.count)
            values = Array(UnsafeBufferPointer(start: pointer, count: multiArray.count)).map(Double.init)
        case .int8:
            let pointer = multiArray.dataPointer.bindMemory(to: Int8.self, capacity: multiArray.count)
            values = Array(UnsafeBufferPointer(start: pointer, count: multiArray.count)).map(Double.init)
        @unknown default:
            throw PULSEOnDeviceError.invalidPrediction("Unsupported MLMultiArray data type.")
        }
        let spatialCount = fetalSpatialElementCount(for: shape)
        let start = max(0, values.count - spatialCount)
        return (values, shape, Array(values[start...]))
    }

    func fetalSpatialDimensions(shape: [Int]) throws -> (width: Int, height: Int) {
        guard shape.count >= 2 else {
            throw PULSEOnDeviceError.invalidPrediction("Invalid spatial tensor shape.")
        }
        let width = shape[shape.count - 1]
        let height = shape[shape.count - 2]
        guard width > 0, height > 0 else {
            throw PULSEOnDeviceError.invalidPrediction("Invalid spatial tensor dimensions.")
        }
        return (width, height)
    }

    func fetalSpatialElementCount(for shape: [Int]) -> Int {
        guard shape.count >= 2 else { return shape.reduce(1, *) }
        return shape[shape.count - 1] * shape[shape.count - 2]
    }

    func fetalFormat(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}
