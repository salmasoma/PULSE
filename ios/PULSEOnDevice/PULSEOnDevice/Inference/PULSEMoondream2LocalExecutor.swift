import CryptoKit
import Foundation
import UIKit

enum PULSEMoondream2LocalExecutor {
    static var isRuntimeReady: Bool {
        let textModelBaseName = stagedManifest?.textModelFilename.replacingOccurrences(of: ".gguf", with: "") ?? ""
        return Bundle.main.findResource(
            named: textModelBaseName,
            extension: "gguf",
            preferredSubdirectories: ["Reasoning", "Resources/Reasoning"]
        ) != nil && stagedManifest?.runtimeSupported == true
    }

    static func generateVisualGrounding(
        report: PULSEAnalysisReport,
        image: UIImage
    ) async throws -> String {
        try await shared.generateVisualGrounding(report: report, image: image)
    }

    static func answerQuestion(
        report: PULSEAnalysisReport,
        image: UIImage,
        question: String
    ) async throws -> String {
        try await shared.answerQuestion(report: report, image: image, question: question)
    }

    static func prewarm() async {
        await shared.prewarm()
    }

    static func prewarm(image: UIImage) async {
        _ = image
        await shared.prewarm()
    }

    private static let shared = Runtime()

    fileprivate static var stagedManifest: PULSEMoondream2Manifest? {
        PULSEMoondream2Manifest.load()
    }
}

private actor Runtime {
    private struct RuntimeConfiguration {
        let contextLength: Int
        let batchSize: Int
        let imageMaxDimension: CGFloat
    }

    private var bridge: PULSEMoondreamBridge?
    private var configuredTextModelURL: URL?
    private var configuredMMProjURL: URL?
    private var cachedPreparedImageURL: URL?
    private var cachedPreparedImageKey: String?

    func prewarm() async {
        guard let manifest = PULSEMoondream2LocalExecutor.stagedManifest,
              manifest.runtimeSupported == true else {
            return
        }
        do {
            let textModelURL = try requireResource(named: manifest.textModelFilename)
            let mmprojURL = try requireResource(named: manifest.mmprojFilename)
            _ = try preparedBridge(
                textModelURL: textModelURL,
                mmprojURL: mmprojURL,
                manifest: manifest
            )
        } catch {
            return
        }
    }

    func prewarm(image: UIImage) async {
        _ = image
        await prewarm()
    }

    func generateVisualGrounding(report: PULSEAnalysisReport, image: UIImage) async throws -> String {
        let manifest = try requireManifest()
        let textModelURL = try requireResource(named: manifest.textModelFilename)
        let mmprojURL = try requireResource(named: manifest.mmprojFilename)
        let bridge = try preparedBridge(
            textModelURL: textModelURL,
            mmprojURL: mmprojURL,
            manifest: manifest
        )
        let preparedImage = try prepareImageForVQA(image, manifest: manifest)

        let output = try bridge.generate(
            withPrompt: makePrompt(report: report),
            imagePath: preparedImage.url.path,
            maxTokens: 48,
            temperature: 0.0,
            topK: 40,
            topP: 0.9
        )

        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4101,
                userInfo: [NSLocalizedDescriptionKey: "The local VQA model returned an empty response."]
            )
        }

        return trimmed
    }

    func answerQuestion(report: PULSEAnalysisReport, image: UIImage, question: String) async throws -> String {
        let cleanedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard cleanedQuestion.isEmpty == false else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4107,
                userInfo: [NSLocalizedDescriptionKey: "A question is required for image-based Q&A."]
            )
        }

        let manifest = try requireManifest()
        let textModelURL = try requireResource(named: manifest.textModelFilename)
        let mmprojURL = try requireResource(named: manifest.mmprojFilename)
        let bridge = try preparedBridge(
            textModelURL: textModelURL,
            mmprojURL: mmprojURL,
            manifest: manifest
        )
        let preparedImage = try prepareImageForVQA(image, manifest: manifest)

        let output = try bridge.generate(
            withPrompt: makeQuestionPrompt(report: report, question: cleanedQuestion),
            imagePath: preparedImage.url.path,
            maxTokens: 40,
            temperature: 0.0,
            topK: 40,
            topP: 0.9
        )

        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4108,
                userInfo: [NSLocalizedDescriptionKey: "The local VQA model returned an empty answer for this question."]
            )
        }

        let cleaned = sanitizeVQAAnswer(trimmed)
        guard cleaned.isEmpty == false, isGenericAssistantAnswer(cleaned) == false else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4109,
                userInfo: [NSLocalizedDescriptionKey: "The local VQA model returned a generic non-answer for this question."]
            )
        }

        guard isContextCompatibleAnswer(cleaned, question: cleanedQuestion, report: report) else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4110,
                userInfo: [NSLocalizedDescriptionKey: "The local VQA answer was inconsistent with the current structured findings for this anatomy question."]
            )
        }

        guard isNonDegenerateAnswer(cleaned, question: cleanedQuestion) else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4111,
                userInfo: [NSLocalizedDescriptionKey: "The local VQA answer was too generic for this question."]
            )
        }

        return cleaned
    }

    private func preparedBridge(
        textModelURL: URL,
        mmprojURL: URL,
        manifest: PULSEMoondream2Manifest
    ) throws -> PULSEMoondreamBridge {
        if
            let bridge,
            configuredTextModelURL == textModelURL,
            configuredMMProjURL == mmprojURL
        {
            return bridge
        }

        let config = runtimeConfiguration(for: manifest)
        let newBridge = try PULSEMoondreamBridge(
            modelPath: textModelURL.path,
            mmprojPath: mmprojURL.path,
            contextLength: config.contextLength,
            batchSize: config.batchSize,
            threads: recommendedThreadCount(),
            useMetal: true
        )

        bridge = newBridge
        configuredTextModelURL = textModelURL
        configuredMMProjURL = mmprojURL
        return newBridge
    }

    private func requireManifest() throws -> PULSEMoondream2Manifest {
        guard let manifest = PULSEMoondream2Manifest.load() else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4103,
                userInfo: [NSLocalizedDescriptionKey: "Local VQA assets were not staged into the app bundle."]
            )
        }
        return manifest
    }

    private func requireResource(named filename: String?) throws -> URL {
        guard let filename, !filename.isEmpty else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4104,
                userInfo: [NSLocalizedDescriptionKey: "The local VQA manifest is missing a required filename."]
            )
        }

        let name = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        guard
            let url = Bundle.main.findResource(
                named: name,
                extension: ext,
                preferredSubdirectories: ["Reasoning", "Resources/Reasoning"]
            )
        else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4105,
                userInfo: [NSLocalizedDescriptionKey: "Missing bundled local VQA asset: \(filename)."]
            )
        }
        return url
    }

    private func prepareImageForVQA(
        _ image: UIImage,
        manifest: PULSEMoondream2Manifest
    ) throws -> (url: URL, cacheKey: String) {
        let config = runtimeConfiguration(for: manifest)
        let preparedImage = resizedImageIfNeeded(image, maxDimension: config.imageMaxDimension)
        guard let jpegData = preparedImage.jpegData(compressionQuality: 0.76) else {
            throw NSError(
                domain: "PULSEMoondream2LocalExecutor",
                code: 4106,
                userInfo: [NSLocalizedDescriptionKey: "Failed to encode the ultrasound image for local VQA."]
            )
        }

        let cacheKey = Insecure.MD5.hash(data: jpegData)
            .map { String(format: "%02hhx", $0) }
            .joined()

        if let cachedPreparedImageURL,
           cachedPreparedImageKey == cacheKey,
           FileManager.default.fileExists(atPath: cachedPreparedImageURL.path) {
            return (cachedPreparedImageURL, cacheKey)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pulse_local_vqa_\(cacheKey)")
            .appendingPathExtension("jpg")

        if !FileManager.default.fileExists(atPath: url.path) {
            try jpegData.write(to: url, options: .atomic)
        }

        cachedPreparedImageURL = url
        cachedPreparedImageKey = cacheKey
        return (url, cacheKey)
    }

    private func runtimeConfiguration(for manifest: PULSEMoondream2Manifest) -> RuntimeConfiguration {
        switch manifest.provider.lowercased() {
        case "radiology_infer_mini":
            // Qwen2-VL can expand large frames into thousands of visual tokens.
            // Keep the image closer to the native 560px vision size and give the
            // text context more room so image decode does not exhaust KV slots.
            return RuntimeConfiguration(
                contextLength: 4096,
                batchSize: 32,
                imageMaxDimension: 560
            )
        case "medix_r1":
            return RuntimeConfiguration(
                contextLength: 2048,
                batchSize: 32,
                imageMaxDimension: 560
            )
        default:
            return RuntimeConfiguration(
                contextLength: 2048,
                batchSize: 64,
                imageMaxDimension: 896
            )
        }
    }

    private func makePrompt(report: PULSEAnalysisReport) -> String {
        """
        You are reviewing an ultrasound frame. Return exactly three lines.

        VISUAL_OBSERVATION: 1 to 3 concise sentences describing only visible sonographic features. Do not diagnose. Do not mention UI, timestamps, or image color scheme.
        IMAGE_ONLY_ORGAN_VIEW: the most likely organ or view from the image alone, or "uncertain" if not clear.
        VISUAL_CERTAINTY: high, medium, or low.

        The image is already provided. Return only those three labeled lines. If anatomy is unclear, say so briefly and set VISUAL_CERTAINTY accordingly. This is a \(report.detectedDomain.pulseDisplayText.lowercased()) ultrasound case, but do not copy automated labels unless they are directly visible.
        """
    }

    private func makeQuestionPrompt(report: PULSEAnalysisReport, question: String) -> String {
        let primary = (report.subviewText ?? report.detectedLabel).pulseDisplayText
        let loweredQuestion = question.lowercased()
        let shouldAttachSupportContext =
            loweredQuestion.contains("measurement")
            || loweredQuestion.contains("length")
            || loweredQuestion.contains("size")
            || loweredQuestion.contains("dimension")
            || loweredQuestion.contains("segmentation")
            || loweredQuestion.contains("mask")
            || loweredQuestion.contains("lesion")
            || loweredQuestion.contains("area")
            || loweredQuestion.contains("volume")
            || loweredQuestion.contains("confidence")
            || loweredQuestion.contains("quality")
        let supportContext = shouldAttachSupportContext
            ? report.specialistFindings
            .filter { $0.category == .measurement || $0.category == .segmentation }
            .prefix(3)
            .map { finding in
                let details = finding.displayDetails.prefix(3).joined(separator: "; ")
                if details.isEmpty {
                    return "- \(finding.title): \(finding.summary)"
                }
                return "- \(finding.title): \(finding.summary) \(details)"
            }
            .joined(separator: "\n")
            : ""

        let optionalContext = supportContext.isEmpty ? "" : "\nOptional structured support context:\n\(supportContext)\n"

        return """
        Answer this ultrasound question directly from the image.

        Question: \(question)

        Rules:
        - Use the image first.
        - Use the optional structured context only if it helps answer the question.
        - Do not give a generic caption.
        - Do not add a follow-up question or assistant-closing phrase.
        - If the answer is uncertain from this single frame, say so briefly.
        - Keep the answer to 1 or 2 concise sentences.

        Case context:
        - Domain: \(report.detectedDomain.pulseDisplayText)
        - Primary result: \(primary)
        \(optionalContext)
        Answer:
        """
    }

    private func recommendedThreadCount() -> Int {
        let active = ProcessInfo.processInfo.activeProcessorCount
        return min(max(active / 2, 4), 6)
    }

    private func resizedImageIfNeeded(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
        let currentMax = max(image.size.width, image.size.height)
        guard currentMax > maxDimension, currentMax > 0 else {
            return image
        }

        let scale = maxDimension / currentMax
        let targetSize = CGSize(
            width: floor(image.size.width * scale),
            height: floor(image.size.height * scale)
        )
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    private func sanitizeVQAAnswer(_ answer: String) -> String {
        var cleaned = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        let closers = [
            "is there anything else i can help you with",
            "is there anything else i can assist you with",
            "let me know if you have any other questions",
            "let me know if you need anything else",
            "how else can i help"
        ]

        for closer in closers {
            if let range = cleaned.range(of: closer, options: [.caseInsensitive]) {
                cleaned = String(cleaned[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        return cleaned
    }

    private func isGenericAssistantAnswer(_ answer: String) -> Bool {
        let lowered = answer.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let genericAnswers = [
            "is there anything else i can help you with",
            "is there anything else i can assist you with",
            "how can i help you",
            "how may i help you",
            "let me know if you have any other questions"
        ]
        return genericAnswers.contains(lowered)
    }

    private func isContextCompatibleAnswer(_ answer: String, question: String, report: PULSEAnalysisReport) -> Bool {
        let normalizedQuestion = normalizedTokens(question)
        let anatomyQuestionTokens: Set<String> = [
            "organ", "view", "structure", "shown", "showing", "anatomy", "plane", "subview"
        ]

        guard normalizedQuestion.intersection(anatomyQuestionTokens).isEmpty == false else {
            return true
        }

        let normalizedAnswer = normalizedTokens(answer)
        guard normalizedAnswer.isEmpty == false else {
            return false
        }

        let primary = (report.subviewText ?? report.detectedLabel).pulseDisplayText
        let expectedTokens = normalizedTokens(primary)
            .union(normalizedTokens(report.detectedDomain.pulseDisplayText))
            .subtracting(["image", "ultrasound", "view"])

        guard expectedTokens.isEmpty == false else {
            return true
        }

        return normalizedAnswer.intersection(expectedTokens).isEmpty == false
    }

    private func isNonDegenerateAnswer(_ answer: String, question: String) -> Bool {
        let loweredAnswer = answer.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard loweredAnswer.isEmpty == false else {
            return false
        }

        let answerTokens = loweredAnswer
            .components(separatedBy: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
            .filter { $0.isEmpty == false }

        guard let firstToken = answerTokens.first else {
            return false
        }

        let hasBinaryLead = firstToken == "yes" || firstToken == "no"
        let binaryQuestion = isBinaryQuestion(question)

        if binaryQuestion {
            if answerTokens.count == 1 {
                return false
            }
            if hasBinaryLead == false {
                return true
            }
            return answerTokens.count >= 4 || loweredAnswer.contains("because") || loweredAnswer.contains("based on")
        }

        return hasBinaryLead == false
    }

    private func isBinaryQuestion(_ question: String) -> Bool {
        let loweredQuestion = question.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let binaryPrefixes = [
            "is ", "are ", "does ", "do ", "did ",
            "can ", "could ", "would ", "will ",
            "was ", "were ", "has ", "have ", "had "
        ]
        return binaryPrefixes.contains { loweredQuestion.hasPrefix($0) }
    }

    private func normalizedTokens(_ text: String) -> Set<String> {
        let separators = CharacterSet.alphanumerics.inverted
        return Set(
            text.lowercased()
                .components(separatedBy: separators)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { token in
                    token.count >= 2 &&
                    !["the", "and", "for", "with", "this", "that", "from", "image"].contains(token)
                }
        )
    }
}
