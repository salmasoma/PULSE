import CoreML
import Foundation
import UIKit

@MainActor
final class PULSEOnDevicePipeline: ObservableObject {
    @Published var isRunning = false
    @Published var lastError: String?
    @Published var lastReport: PULSEAnalysisReport?

    private let registry = PULSEModelRegistry.shared

    func analyze(primaryImage: UIImage, prompt: String = "", extraImages: [String: UIImage] = [:]) async {
        do {
            _ = try await analyzeReport(primaryImage: primaryImage, prompt: prompt, extraImages: extraImages)
        } catch {
            lastReport = nil
            lastError = error.localizedDescription
        }
    }

    func analyzeReport(primaryImage: UIImage, prompt: String = "", extraImages: [String: UIImage] = [:]) async throws -> PULSEAnalysisReport {
        isRunning = true
        lastError = nil
        defer { isRunning = false }

        do {
            let report = try await buildReport(primaryImage: primaryImage, prompt: prompt, extraImages: extraImages)
            lastReport = report
            return report
        } catch {
            lastReport = nil
            lastError = error.localizedDescription
            throw error
        }
    }

    private func buildReport(primaryImage: UIImage, prompt: String, extraImages: [String: UIImage]) async throws -> PULSEAnalysisReport {
        let perceptionStart = DispatchTime.now().uptimeNanoseconds
        var stageTimings: [PULSEStageTiming] = []

        let qualityStart = DispatchTime.now().uptimeNanoseconds
        let quality = qualitySummary(for: primaryImage)
        stageTimings.append(
            PULSEStageTiming(
                stageID: "quality_assessment",
                title: "Quality assessment",
                elapsedMs: elapsedMs(since: qualityStart)
            )
        )

        guard let routerDescriptor = await registry.modelDescriptor(taskID: "system/domain_classification") else {
            throw PULSEOnDeviceError.invalidPrediction("The on-device router is missing from the manifest.")
        }
        let routingStart = DispatchTime.now().uptimeNanoseconds
        let routingPrediction = try await predict(descriptor: routerDescriptor, primaryImage: primaryImage, extraImages: extraImages)
        stageTimings.append(
            PULSEStageTiming(
                stageID: routerDescriptor.taskID,
                title: "Domain router",
                elapsedMs: elapsedMs(since: routingStart),
                count: 1
            )
        )
        let domainLabel = routingPrediction.label ?? "unknown"

        let descriptors = await registry.descriptors(for: domainLabel).filter { $0.runtimeEnabled }

        var findings = [PULSEFinding]()
        var detectedLabel = routingPrediction.label ?? domainLabel
        findings.append(
            PULSEFinding(
                taskID: routerDescriptor.taskID,
                title: "Domain",
                summary: "Detected domain: \(domainLabel.pulseDisplayText).",
                label: domainLabel,
                confidence: routingPrediction.confidence,
                details: routingPrediction.details,
                artifacts: routingPrediction.artifacts
            )
        )

        if domainLabel == "fetal" {
            let fetalStart = DispatchTime.now().uptimeNanoseconds
            let fetalSpecialists = try await runFetalSpecialists(primaryImage: primaryImage)
            findings.append(contentsOf: fetalSpecialists.findings)
            stageTimings.append(
                PULSEStageTiming(
                    stageID: "fetal_specialists",
                    title: "Fetal specialist cascade",
                    elapsedMs: elapsedMs(since: fetalStart),
                    count: fetalSpecialists.findings.count
                )
            )
            if let label = fetalSpecialists.detectedLabel {
                detectedLabel = label
            }
        } else {
            for descriptor in descriptors {
                if descriptor.taskType == "multimodal" {
                    let required = Set(descriptor.modalities.filter { $0 != "bmode" && $0 != "image" })
                    if !required.isSubset(of: Set(extraImages.keys)) {
                        continue
                    }
                }
                let specialistStart = DispatchTime.now().uptimeNanoseconds
                let prediction = try await predict(descriptor: descriptor, primaryImage: primaryImage, extraImages: extraImages)
                findings.append(
                    PULSEFinding(
                        taskID: descriptor.taskID,
                        title: userFacingTitle(for: descriptor),
                        summary: prediction.summary,
                        label: prediction.label,
                        confidence: prediction.confidence,
                        details: prediction.details,
                        artifacts: prediction.artifacts
                    )
                )
                stageTimings.append(
                    PULSEStageTiming(
                        stageID: descriptor.taskID,
                        title: userFacingTitle(for: descriptor),
                        elapsedMs: elapsedMs(since: specialistStart),
                        count: 1
                    )
                )
            }
        }

        let note = prompt.isEmpty
            ? "Prompt-free on-device execution. Structured perception and reasoning are generated locally from bundled models and on-device report logic."
            : "Prompt recorded locally. The prompt is used only to focus the on-device reasoning report and does not change the underlying perception outputs."
        let perceptionMs = elapsedMs(since: perceptionStart)
        let profiling = PULSEProfilingSummary(
            deviceIdentifier: currentDeviceIdentifier(),
            perceptionMs: perceptionMs,
            reasoningMs: nil,
            endToEndMs: perceptionMs,
            stages: stageTimings
        )
        return PULSEAnalysisReport(
            createdAt: Date(),
            detectedDomain: domainLabel,
            detectedLabel: detectedLabel,
            quality: quality,
            findings: findings,
            note: note,
            profiling: profiling
        )
    }

    private func predict(
        descriptor: PULSECoreMLModelDescriptor,
        primaryImage: UIImage,
        extraImages: [String: UIImage]
    ) async throws -> (label: String?, confidence: Double?, summary: String, details: [String], artifacts: [PULSEArtifactImage]) {
        let model = try await registry.loadModel(for: descriptor)
        let provider = try featureProvider(for: descriptor, primaryImage: primaryImage, extraImages: extraImages)
        let output = try await model.prediction(from: provider)
        return try decode(output: output, descriptor: descriptor, primaryImage: primaryImage)
    }

    private func featureProvider(
        for descriptor: PULSECoreMLModelDescriptor,
        primaryImage: UIImage,
        extraImages: [String: UIImage]
    ) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [:]
        for inputName in descriptor.inputNames {
            let image: UIImage
            if inputName == "image" || inputName == "bmode" {
                image = primaryImage
            } else if let extra = extraImages[inputName] {
                image = extra
            } else {
                throw PULSEOnDeviceError.invalidPrediction("Missing required modality `\(inputName)` for `\(descriptor.taskID)`.")
            }
            let pixelBuffer = try ImageBufferFactory.pixelBuffer(from: image, size: descriptor.imageSize)
            features[inputName] = MLFeatureValue(pixelBuffer: pixelBuffer)
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func decode(
        output: MLFeatureProvider,
        descriptor: PULSECoreMLModelDescriptor,
        primaryImage: UIImage
    ) throws -> (label: String?, confidence: Double?, summary: String, details: [String], artifacts: [PULSEArtifactImage]) {
        guard let outputName = descriptor.outputNames.first,
              let value = output.featureValue(for: outputName)
        else {
            throw PULSEOnDeviceError.invalidPrediction("Missing output `\(descriptor.outputNames.joined(separator: ", "))` for `\(descriptor.taskID)`.")
        }

        switch descriptor.outputSemantics {
        case "logits":
            let logits = try flattenedArray(from: value)
            let probabilities = softmax(logits)
            let bestIndex = probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            let label = descriptor.labels.indices.contains(bestIndex) ? descriptor.labels[bestIndex] : "class_\(bestIndex)"
            let confidence = probabilities.indices.contains(bestIndex) ? probabilities[bestIndex] : 0.0
            return (label, confidence, classificationSummary(label: label, descriptor: descriptor), [], [])

        case "binary_mask_logits":
            let tensor = try tensorData(from: value)
            let spatial = try spatialDimensions(from: tensor.shape)
            let logits = tensor.spatialSliceValues
            let positiveFraction = logits.map(sigmoid).filter { $0 > 0.5 }.count
            let fraction = Double(positiveFraction) / Double(max(logits.count, 1))
            let mask = logits.map { sigmoid($0) > 0.5 ? UInt8(1) : UInt8(0) }
            let maskMetrics = binaryMaskMetrics(mask: mask, width: spatial.width, height: spatial.height)
            let artifacts = makeBinarySegmentationArtifacts(
                mask: mask,
                width: spatial.width,
                height: spatial.height,
                descriptor: descriptor,
                primaryImage: primaryImage
            )
            var details = ["positive_fraction: \(format(fraction))"]
            details.append("segmented_area_px: \(maskMetrics.areaPixels)")
            if let bbox = maskMetrics.boundingBox {
                details.append("bbox_width_px: \(bbox.width)")
                details.append("bbox_height_px: \(bbox.height)")
                details.append("max_dimension_px: \(max(bbox.width, bbox.height))")
            }
            return (nil, nil, "Segmentation generated for the selected region.", details, artifacts)

        case "mask_class_logits":
            let tensor = try tensorData(from: value)
            let parsed = try multiClassMask(from: tensor, descriptor: descriptor)
            let maskMetrics = multiClassMaskMetrics(mask: parsed.mask, width: parsed.width, height: parsed.height)
            let classFractions = parsed.classFractions
                .filter { $0.value > 0 }
                .map { "\($0.key): \(format($0.value))" }
                .sorted()
            let artifacts = makeMultiClassSegmentationArtifacts(
                mask: parsed.mask,
                width: parsed.width,
                height: parsed.height,
                descriptor: descriptor,
                primaryImage: primaryImage
            )
            var details = classFractions.isEmpty ? ["class_count: \(parsed.classCount)"] : classFractions
            details.append("segmented_area_px: \(maskMetrics.areaPixels)")
            if let bbox = maskMetrics.boundingBox {
                details.append("bbox_width_px: \(bbox.width)")
                details.append("bbox_height_px: \(bbox.height)")
                details.append("max_dimension_px: \(max(bbox.width, bbox.height))")
            }
            return (nil, nil, "Segmentation generated for multiple regions.", details, artifacts)

        case "objectness_cxcywh":
            let values = try flattenedArray(from: value)
            guard values.count >= 5 else {
                throw PULSEOnDeviceError.invalidPrediction("Detection output is malformed for `\(descriptor.taskID)`.")
            }
            let objectness = sigmoid(values[0])
            return (
                nil,
                objectness,
                "Detection head produced a bounding-box proposal.",
                [
                    "cx: \(format(values[1]))",
                    "cy: \(format(values[2]))",
                    "w: \(format(values[3]))",
                    "h: \(format(values[4]))"
                ],
                []
            )

        case "scalar_regression":
            let values = try flattenedArray(from: value)
            let scalar = values.first ?? 0.0
            return (nil, nil, "Measurement generated.", ["value: \(format(scalar))"], [])

        default:
            throw PULSEOnDeviceError.invalidPrediction("Unsupported output semantics `\(descriptor.outputSemantics)`.")
        }
    }

    private func qualitySummary(for image: UIImage) -> PULSEQualitySummary {
        let stats = ImageBufferFactory.grayscaleStatistics(for: image)
        let brightness = stats?.brightness ?? 0.0
        let contrast = stats?.contrast ?? 0.0
        let label: String
        if brightness < 0.08 || contrast < 0.05 {
            label = "low"
        } else if brightness < 0.16 || contrast < 0.09 {
            label = "medium"
        } else {
            label = "good"
        }
        return PULSEQualitySummary(brightness: brightness, contrast: contrast, qualityLabel: label)
    }

    private func elapsedMs(since startUptimeNs: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - startUptimeNs) / 1_000_000.0
    }

    private func currentDeviceIdentifier() -> String {
        #if targetEnvironment(simulator)
        return "iOS-Simulator"
        #else
        var systemInfo = utsname()
        uname(&systemInfo)
        let identifier = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(cString: $0)
            }
        }
        return identifier
        #endif
    }

    private func flattenedArray(from featureValue: MLFeatureValue) throws -> [Double] {
        try tensorData(from: featureValue).values
    }

    private func userFacingTitle(for descriptor: PULSECoreMLModelDescriptor) -> String {
        let task = descriptor.taskID.lowercased()
        if task.contains("view_classification") {
            return "View"
        }
        if task.contains("organ_classification") {
            return "Organ"
        }
        if task.contains("anomaly_detection") {
            return "Anomaly Screening"
        }
        if task.contains("nodule_detection") || task.contains("detection") {
            return "Localization"
        }
        if task.contains("segmentation") {
            return "Segmentation"
        }
        if task.contains("imt") || task.contains("measurement") || task.contains("regression") {
            return "Measurement"
        }
        if task.contains("multimodal") {
            return "Integrated Assessment"
        }
        if task.contains("classification") {
            return "Assessment"
        }
        return descriptor.title
    }

    private func classificationSummary(label: String, descriptor: PULSECoreMLModelDescriptor) -> String {
        let display = label.pulseTrimmed(for: descriptor.domain).pulseDisplayText
        let task = descriptor.taskID.lowercased()
        if task.contains("view_classification") {
            return "Detected view: \(display)."
        }
        if task.contains("organ_classification") {
            return "Detected organ: \(display)."
        }
        if task.contains("anomaly_detection") {
            return "Anomaly screening result: \(display)."
        }
        if task.contains("nodule_detection") {
            return "Localized finding: \(display)."
        }
        return "Primary result: \(display)."
    }

    private func tensorData(from featureValue: MLFeatureValue) throws -> (values: [Double], shape: [Int], spatialSliceValues: [Double]) {
        if let multiArray = featureValue.multiArrayValue {
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
            let spatialSliceStart = max(0, values.count - spatialElementCount(for: shape))
            return (values, shape, Array(values[spatialSliceStart...]))
        }
        if featureValue.type == .double {
            let number = featureValue.doubleValue
            return ([number], [1], [number])
        }
        throw PULSEOnDeviceError.invalidPrediction("Output is not an MLMultiArray or scalar double.")
    }

    private func spatialDimensions(from shape: [Int]) throws -> (width: Int, height: Int) {
        if shape.count >= 2 {
            let width = shape[shape.count - 1]
            let height = shape[shape.count - 2]
            if width > 0 && height > 0 {
                return (width, height)
            }
        }
        throw PULSEOnDeviceError.invalidPrediction("Unable to determine segmentation output dimensions.")
    }

    private func spatialElementCount(for shape: [Int]) -> Int {
        guard shape.count >= 2 else { return shape.reduce(1, *) }
        return shape[shape.count - 1] * shape[shape.count - 2]
    }

    private func multiClassMask(
        from tensor: (values: [Double], shape: [Int], spatialSliceValues: [Double]),
        descriptor: PULSECoreMLModelDescriptor
    ) throws -> (mask: [UInt8], width: Int, height: Int, classCount: Int, classFractions: [String: Double]) {
        let shape = tensor.shape
        let width: Int
        let height: Int
        let channels: Int

        switch shape.count {
        case 4:
            channels = shape[1]
            height = shape[2]
            width = shape[3]
        case 3:
            channels = shape[0]
            height = shape[1]
            width = shape[2]
        default:
            throw PULSEOnDeviceError.invalidPrediction("Unexpected multi-class segmentation tensor shape.")
        }

        let planeSize = width * height
        guard planeSize > 0, channels > 0 else {
            throw PULSEOnDeviceError.invalidPrediction("Segmentation tensor is empty.")
        }

        let startIndex = max(0, tensor.values.count - (channels * planeSize))
        let values = Array(tensor.values[startIndex...])
        var mask = [UInt8](repeating: 0, count: planeSize)

        for pixelIndex in 0..<planeSize {
            var bestClass = 0
            var bestLogit = -Double.infinity
            for classIndex in 0..<channels {
                let value = values[(classIndex * planeSize) + pixelIndex]
                if value > bestLogit {
                    bestLogit = value
                    bestClass = classIndex
                }
            }
            mask[pixelIndex] = UInt8(bestClass)
        }

        let labels = descriptorLabelsForMaskChannels(channelCount: channels, descriptor: descriptor)
        let total = Double(max(planeSize, 1))
        var fractions: [String: Double] = [:]
        for classIndex in 1..<channels {
            let count = mask.reduce(0) { partial, value in
                partial + (value == UInt8(classIndex) ? 1 : 0)
            }
            let fraction = Double(count) / total
            if fraction > 0 {
                let label = labels.indices.contains(classIndex) ? labels[classIndex] : "class_\(classIndex)"
                fractions[label] = fraction
            }
        }

        return (mask, width, height, channels, fractions)
    }

    private func descriptorLabelsForMaskChannels(channelCount: Int, descriptor: PULSECoreMLModelDescriptor) -> [String] {
        if channelCount == descriptor.labels.count + 1 {
            return ["background"] + descriptor.labels
        }
        if channelCount == descriptor.labels.count {
            return descriptor.labels
        }
        let generated = (0..<channelCount).map { index in
            index == 0 ? "background" : "class_\(index)"
        }
        return generated
    }

    private func binaryMaskMetrics(mask: [UInt8], width: Int, height: Int) -> (areaPixels: Int, boundingBox: CGRect?) {
        let areaPixels = mask.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
        return (areaPixels, foregroundBoundingBox(mask: mask, width: width, height: height))
    }

    private func multiClassMaskMetrics(mask: [UInt8], width: Int, height: Int) -> (areaPixels: Int, boundingBox: CGRect?) {
        let areaPixels = mask.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
        return (areaPixels, foregroundBoundingBox(mask: mask, width: width, height: height))
    }

    private func foregroundBoundingBox(mask: [UInt8], width: Int, height: Int) -> CGRect? {
        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard mask[index] > 0 else { continue }
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }

        guard maxX >= 0, maxY >= 0 else { return nil }
        return CGRect(
            x: minX,
            y: minY,
            width: maxX - minX + 1,
            height: maxY - minY + 1
        )
    }

    private func makeBinarySegmentationArtifacts(
        mask: [UInt8],
        width: Int,
        height: Int,
        descriptor: PULSECoreMLModelDescriptor,
        primaryImage: UIImage
    ) -> [PULSEArtifactImage] {
        let color = artifactColors(for: descriptor).first ?? UIColor.systemTeal
        guard
            let maskImage = makeMaskImage(mask: mask, width: width, height: height, colors: [UIColor.clear, color], background: UIColor(red: 0.04, green: 0.09, blue: 0.16, alpha: 1.0)),
            let overlayImage = makeOverlayImage(mask: mask, width: width, height: height, colors: [UIColor.clear, color.withAlphaComponent(0.62)], primaryImage: primaryImage, modelSize: width),
            let maskData = maskImage.pngData(),
            let overlayData = overlayImage.pngData()
        else {
            return []
        }

        return [
            PULSEArtifactImage(kind: "overlay", caption: "Segmentation overlay", pngData: overlayData),
            PULSEArtifactImage(kind: "mask", caption: "Segmentation mask", pngData: maskData),
        ]
    }

    private func makeMultiClassSegmentationArtifacts(
        mask: [UInt8],
        width: Int,
        height: Int,
        descriptor: PULSECoreMLModelDescriptor,
        primaryImage: UIImage
    ) -> [PULSEArtifactImage] {
        let foregroundColors = artifactColors(for: descriptor)
        let colors = [UIColor.clear] + foregroundColors
        let overlayColors = colors.enumerated().map { index, color in
            index == 0 ? UIColor.clear : color.withAlphaComponent(0.58)
        }
        guard
            let maskImage = makeMaskImage(mask: mask, width: width, height: height, colors: colors, background: UIColor(red: 0.04, green: 0.09, blue: 0.16, alpha: 1.0)),
            let overlayImage = makeOverlayImage(mask: mask, width: width, height: height, colors: overlayColors, primaryImage: primaryImage, modelSize: width),
            let maskData = maskImage.pngData(),
            let overlayData = overlayImage.pngData()
        else {
            return []
        }

        return [
            PULSEArtifactImage(kind: "overlay", caption: "Segmentation overlay", pngData: overlayData),
            PULSEArtifactImage(kind: "mask", caption: "Segmentation mask", pngData: maskData),
        ]
    }

    private func artifactColors(for descriptor: PULSECoreMLModelDescriptor) -> [UIColor] {
        switch descriptor.taskID {
        case "breast/lesion_segmentation":
            return [UIColor(red: 0.91, green: 0.43, blue: 0.28, alpha: 1.0)]
        case "carotid/lumen_segmentation":
            return [UIColor(red: 0.08, green: 0.62, blue: 0.82, alpha: 1.0)]
        case "liver/segmentation":
            return [
                UIColor(red: 0.16, green: 0.66, blue: 0.47, alpha: 1.0),
                UIColor(red: 0.86, green: 0.35, blue: 0.30, alpha: 1.0)
            ]
        case "kidney/anatomy_segmentation":
            return [
                UIColor(red: 0.24, green: 0.61, blue: 0.92, alpha: 1.0),
                UIColor(red: 0.16, green: 0.79, blue: 0.60, alpha: 1.0),
                UIColor(red: 0.99, green: 0.66, blue: 0.18, alpha: 1.0),
                UIColor(red: 0.74, green: 0.44, blue: 0.92, alpha: 1.0)
            ]
        default:
            return [
                UIColor(red: 0.09, green: 0.64, blue: 0.78, alpha: 1.0),
                UIColor(red: 0.92, green: 0.43, blue: 0.30, alpha: 1.0),
                UIColor(red: 0.20, green: 0.75, blue: 0.56, alpha: 1.0),
                UIColor(red: 0.92, green: 0.78, blue: 0.20, alpha: 1.0)
            ]
        }
    }

    private func makeMaskImage(mask: [UInt8], width: Int, height: Int, colors: [UIColor], background: UIColor) -> UIImage? {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for index in 0..<(width * height) {
            let classIndex = Int(mask[index])
            let color = classIndex == 0 ? background : colors[min(classIndex, colors.count - 1)]
            let rgba = color.rgbaComponents
            let offset = index * 4
            pixels[offset] = rgba.r
            pixels[offset + 1] = rgba.g
            pixels[offset + 2] = rgba.b
            pixels[offset + 3] = 255
        }
        return makeImage(fromRGBA: pixels, width: width, height: height)
    }

    private func makeOverlayImage(mask: [UInt8], width: Int, height: Int, colors: [UIColor], primaryImage: UIImage, modelSize: Int) -> UIImage? {
        guard let canvas = ImageBufferFactory.modelCanvas(from: primaryImage, size: modelSize) else {
            return nil
        }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        return renderer.image { _ in
            canvas.draw(in: CGRect(x: 0, y: 0, width: width, height: height))

            var pixels = [UInt8](repeating: 0, count: width * height * 4)
            for index in 0..<(width * height) {
                let classIndex = Int(mask[index])
                guard classIndex > 0 else { continue }
                let color = colors[min(classIndex, colors.count - 1)]
                let rgba = color.rgbaComponents
                let offset = index * 4
                pixels[offset] = rgba.r
                pixels[offset + 1] = rgba.g
                pixels[offset + 2] = rgba.b
                pixels[offset + 3] = rgba.a
            }
            if let overlay = makeImage(fromRGBA: pixels, width: width, height: height) {
                overlay.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
            }
        }
    }

    private func makeImage(fromRGBA pixels: [UInt8], width: Int, height: Int) -> UIImage? {
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData) else {
            return nil
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
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

    private func softmax(_ logits: [Double]) -> [Double] {
        let maximum = logits.max() ?? 0.0
        let exps = logits.map { Foundation.exp($0 - maximum) }
        let sum = exps.reduce(0.0, +)
        return exps.map { $0 / max(sum, 1e-12) }
    }

    private func sigmoid(_ value: Double) -> Double {
        1.0 / (1.0 + Foundation.exp(-value))
    }

    private func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}

private extension UIColor {
    var rgbaComponents: (r: UInt8, g: UInt8, b: UInt8, a: UInt8) {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        return (
            UInt8(max(0, min(255, Int(red * 255.0)))),
            UInt8(max(0, min(255, Int(green * 255.0)))),
            UInt8(max(0, min(255, Int(blue * 255.0)))),
            UInt8(max(0, min(255, Int(alpha * 255.0))))
        )
    }
}
