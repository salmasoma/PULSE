import Foundation
import UIKit

struct PULSEReasoningService {
    static let processingStatusMessage = "Generating on-device sonography report..."

    struct Availability {
        let isAvailable: Bool
        let providerLabel: String
        let statusText: String
        let detailText: String
    }

    static var availability: Availability {
        if let manifest = PULSEMoondream2Manifest.load() {
            return Availability(
                isAvailable: true,
                providerLabel: manifest.displayLabel,
                statusText: PULSEMoondream2LocalExecutor.isRuntimeReady
                    ? "\(manifest.displayLabel) is available on device."
                    : "\(manifest.displayLabel) assets are bundled locally, and the app will fall back to the grounded local composer if the VLM cannot start.",
                detailText: availabilityDetailText(for: manifest)
            )
        }

        return Availability(
            isAvailable: true,
            providerLabel: "Grounded Local Composer",
            statusText: "On-device reasoning is available in this app build.",
            detailText: "This build generates a structured local sonography report entirely on device from the PULSE findings."
        )
    }

    static var isConfigured: Bool {
        availability.isAvailable
    }

    static func structuredSelectionRationale(for report: PULSEAnalysisReport) -> String? {
        PULSEStructuredReasoningComposer.selectionRationaleText(report: report)
    }

    static func structuredMeasurementSummary(for report: PULSEAnalysisReport) -> String? {
        PULSEStructuredReasoningComposer.measurementSummaryText(report: report)
    }

    static func structuredSegmentationSummary(for report: PULSEAnalysisReport) -> String? {
        PULSEStructuredReasoningComposer.segmentationSummaryText(report: report)
    }

    static func structuredImpression(for report: PULSEAnalysisReport) -> String {
        PULSEStructuredReasoningComposer.impressionSection(report: report)
    }

    static func structuredFindingsSummary(for report: PULSEAnalysisReport, imageSize: CGSize) -> String {
        PULSEStructuredReasoningComposer.findingsSection(
            report: report,
            imageSize: imageSize,
            visualGrounding: nil
        )
    }

    func generateReasonedReport(
        report: PULSEAnalysisReport,
        primaryImage: UIImage
    ) async throws -> PULSEReasonedReport {
        let targetRuntime = PULSEMoondream2Manifest.load()?.displayLabel ?? "Grounded Local Composer"

        var visualGrounding: PULSEVisualGrounding?
        if Self.shouldUseVisualGrounding(report: report) && PULSEMoondream2LocalExecutor.isRuntimeReady {
            do {
                let rawOutput = try await PULSEMoondream2LocalExecutor.generateVisualGrounding(
                    report: report,
                    image: primaryImage
                )
                visualGrounding = PULSEVisualGroundingParser.parse(rawOutput)
            } catch {
                visualGrounding = nil
            }
        }

        return PULSEStructuredReasoningComposer.compose(
            report: report,
            imageSize: primaryImage.size,
            visualGrounding: visualGrounding,
            targetRuntime: visualGrounding == nil ? "Grounded Local Composer" : "\(targetRuntime) + Grounded Local Composer"
        )
    }

    private static func availabilityDetailText(for manifest: PULSEMoondream2Manifest) -> String {
        var lines: [String] = []
        if let modality = manifest.modality, !modality.isEmpty {
            lines.append("Target modality: \(modality).")
        }
        lines.append("Text model: \(manifest.textModelFilename).")
        lines.append("Vision projector: \(manifest.mmprojFilename).")
        if let runtimeTarget = manifest.runtimeTarget, !runtimeTarget.isEmpty {
            lines.append("Runtime: \(runtimeTarget).")
        }
        if let note = manifest.note, !note.isEmpty {
            lines.append(note)
        }
        lines.append("The final report is generated locally as a sonography-style report from structured PULSE outputs, with the bundled VLM used only for limited visual grounding when its output is usable.")
        return lines.joined(separator: " ")
    }

    private static func shouldUseVisualGrounding(report: PULSEAnalysisReport) -> Bool {
        let quality = report.quality.qualityLabel.lowercased()
        if quality == "low" {
            return false
        }

        if report.detectedDomain.lowercased() == "fetal" {
            return false
        }

        if report.subviewText != nil {
            return false
        }

        if report.specialistFindings.contains(where: { $0.category == .measurement || $0.category == .segmentation }) {
            return false
        }

        if let primary = report.primaryFinding {
            switch primary.category {
            case .classification, .measurement, .segmentation:
                return false
            default:
                break
            }
        }

        return report.specialistFindings.count <= 1
    }
}

struct PULSEVisualGrounding {
    let observation: String?
    let organView: String?
    let certainty: String?

    var isUsable: Bool {
        guard let observation, observation.count >= 80 else { return false }
        let lowered = observation.lowercased()
        let bannedPhrases = [
            "cannot generate a response",
            "cannot analyze",
            "please provide a clear image description",
            "date",
            "time",
            "captured at a specific angle",
            "black and white color scheme"
        ]
        return bannedPhrases.allSatisfy { !lowered.contains($0) }
    }
}

private enum PULSEVisualGroundingParser {
    static func parse(_ raw: String) -> PULSEVisualGrounding? {
        let lines = raw
            .replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var observation: String?
        var organView: String?
        var certainty: String?

        for line in lines {
            if line.lowercased().hasPrefix("visual_observation:") {
                observation = value(of: line)
            } else if line.lowercased().hasPrefix("image_only_organ_view:") {
                organView = value(of: line)
            } else if line.lowercased().hasPrefix("visual_certainty:") {
                certainty = value(of: line)
            }
        }

        let grounding = PULSEVisualGrounding(
            observation: observation,
            organView: organView,
            certainty: certainty
        )

        if grounding.observation == nil && grounding.organView == nil && grounding.certainty == nil {
            return nil
        }
        return grounding
    }

    private static func value(of line: String) -> String {
        line.split(separator: ":", maxSplits: 1).dropFirst().first?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
}

enum PULSEStructuredReasoningComposer {
    static func compose(
        report: PULSEAnalysisReport,
        imageSize: CGSize,
        visualGrounding: PULSEVisualGrounding?,
        targetRuntime: String
    ) -> PULSEReasonedReport {
        let body = composeSonographyReport(report: report, imageSize: imageSize, visualGrounding: visualGrounding)
        return PULSEReasonedReport(
            model: targetRuntime,
            mode: "sonography_report",
            output: body
        )
    }

    private static func composeSonographyReport(
        report: PULSEAnalysisReport,
        imageSize: CGSize,
        visualGrounding: PULSEVisualGrounding?
    ) -> String {
        let sections: [(String, String)] = [
            ("EXAM", examSection(report: report)),
            ("TECHNIQUE", techniqueSection(report: report)),
            ("FINDINGS", findingsSection(report: report, imageSize: imageSize, visualGrounding: visualGrounding)),
            ("IMPRESSION", impressionSection(report: report)),
            ("LIMITATIONS", limitationsSection(report: report))
        ]

        return sections
            .filter { !$0.1.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .map { "\($0.0)\n\($0.1)" }
            .joined(separator: "\n\n")
    }

    private static func examSection(report: PULSEAnalysisReport) -> String {
        let domain = report.detectedDomain.lowercased()
        let primary = primaryResult(report)

        if domain == "fetal" {
            if let primary {
                return "Obstetric ultrasound, targeted fetal \(primary.lowercased()) view."
            }
            return "Obstetric ultrasound."
        }

        if let primary, primary.lowercased() != report.detectedDomain.lowercased() {
            return "\(report.detectedDomain.pulseDisplayText) ultrasound, targeted \(primary.lowercased()) view."
        }

        return "\(report.detectedDomain.pulseDisplayText) ultrasound."
    }

    private static func techniqueSection(report: PULSEAnalysisReport) -> String {
        var line = "Single grayscale sonographic frame submitted for review."
        let quality = report.quality.qualityLabel.lowercased()
        if quality.isEmpty == false && quality != "unknown" {
            line += " Image quality is \(quality)."
        }
        return line
    }

    static func findingsSection(
        report: PULSEAnalysisReport,
        imageSize: CGSize,
        visualGrounding: PULSEVisualGrounding?
    ) -> String {
        if isFetalBiometryCase(report) {
            let primary = (primaryResult(report) ?? "targeted fetal subview").lowercased()
            var findings: [String] = [
                "Submitted image is a focused fetal \(primary) biometry frame rather than a whole-fetus survey image."
            ]

            if primary.contains("femur") {
                findings.append("A single dominant elongated echogenic long-bone structure is identified within the field of view and serves as the measurement target.")
            } else if primary.contains("head") || primary.contains("brain") {
                findings.append("Submitted frame is centered on the fetal head with anatomy positioned for targeted biometric assessment rather than general survey imaging.")
            } else if primary.contains("abdomen") {
                findings.append("Submitted frame is centered on the fetal abdomen with anatomy positioned for targeted biometric assessment rather than general survey imaging.")
            } else if primary.contains("cervix") {
                findings.append("Submitted frame is centered on the cervix with limited surrounding anatomy visible on this single image.")
            } else {
                findings.append("Submitted frame is focused on the labeled fetal subview with limited surrounding anatomy visible on this single image.")
            }

            if let measurementText = measurementSummaryText(report: report) {
                findings.append(measurementText)
            }

            if let segmentationText = segmentationSummaryText(report: report) {
                findings.append(segmentationText)
            }

            findings.append("No gross additional morphologic abnormality can be established confidently from this isolated still image.")
            return findings.joined(separator: " ")
        }

        var parts: [String] = []
        let visual = visualDescription(report: report, imageSize: imageSize, visualGrounding: visualGrounding)
        if visual.isEmpty == false {
            parts.append(visual)
        }

        if let pathology = primaryPathologyFinding(in: report), let label = pathology.label?.pulseDisplayText {
            var pathologySentence = "Structured classification favors \(label.lowercased())"
            if let confidence = pathology.confidencePercentText {
                pathologySentence += " at \(confidence)"
            }
            pathologySentence += "."
            parts.append(pathologySentence)
        } else if let primary = primaryResult(report) {
            parts.append("Structured analysis localizes the frame to \(report.detectedDomain.pulseDisplayText.lowercased()) anatomy with primary interpretation of \(primary.lowercased()).")
        }

        if let measurementText = measurementSummaryText(report: report) {
            parts.append(measurementText)
        }

        if let segmentationText = segmentationSummaryText(report: report) {
            parts.append(segmentationText)
        }

        if let rationaleText = selectionRationaleText(report: report) {
            parts.append(rationaleText)
        }

        return parts.joined(separator: " ")
    }

    static func impressionSection(report: PULSEAnalysisReport) -> String {
        var lines: [String] = []

        if isFetalBiometryCase(report) {
            let primary = (primaryResult(report) ?? "targeted fetal subview").lowercased()
            lines.append("Targeted fetal \(primary) view.")
            if let label = measurementLabel(report: report) {
                lines.append("\(label) measurement successfully obtained, supporting appropriate subview acquisition.")
            }
            lines.append("Single submitted image is suitable for subview identification and biometry correlation but is not sufficient for definitive fetal anomaly assessment.")
            return numbered(lines)
        }

        if let pathology = primaryPathologyFinding(in: report), let label = pathology.label?.pulseDisplayText {
            var sentence = "Primary structured assessment favors \(label.lowercased())"
            if let confidence = pathology.confidencePercentText {
                sentence += " at \(confidence)"
            }
            sentence += "."
            lines.append(sentence)
            if let rationaleLine = impressionRationaleLine(report: report) {
                lines.append(rationaleLine)
            }
            lines.append("Interpretation should be correlated with the complete examination and supporting clinical context.")
            return numbered(lines)
        }

        if let primary = primaryResult(report) {
            lines.append("Primary interpretation is \(primary.lowercased()) within the \(report.detectedDomain.pulseDisplayText.lowercased()) domain.")
            if let rationaleLine = impressionRationaleLine(report: report) {
                lines.append(rationaleLine)
            }
            if let label = measurementLabel(report: report) {
                lines.append("\(label) measurement was obtained on the submitted frame.")
            }
            if let segmentationImpression = segmentationImpressionText(report: report) {
                lines.append(segmentationImpression)
            }
            lines.append("This image is more supportive of view identification than of definitive pathology assessment from a single still frame.")
            return numbered(lines)
        }

        lines.append("Single submitted image supports \(report.detectedDomain.pulseDisplayText.lowercased()) domain routing only.")
        lines.append("No definitive pathology assessment should be made from this frame in isolation.")
        return numbered(lines)
    }

    private static func limitationsSection(report: PULSEAnalysisReport) -> String {
        var lines = ["Interpretation is limited by review of a single still image rather than the complete cine examination."]
        let quality = report.quality.qualityLabel.lowercased()
        if quality.isEmpty == false && quality != "unknown" {
            lines.append("Image quality is \(quality), which further limits assessment.")
        }
        if isFetalBiometryCase(report) {
            lines.append("Absence of an obvious abnormality on this frame does not exclude abnormality elsewhere in the examination.")
        }
        return lines.joined(separator: " ")
    }

    private static func visualDescription(
        report: PULSEAnalysisReport,
        imageSize: CGSize,
        visualGrounding: PULSEVisualGrounding?
    ) -> String {
        if let visualGrounding, visualGrounding.isUsable, let observation = visualGrounding.observation {
            return observation
        }
        return fallbackVisualDescription(report: report, imageSize: imageSize)
    }

    private static func fallbackVisualDescription(report: PULSEAnalysisReport, imageSize: CGSize) -> String {
        let result = (primaryResult(report) ?? report.detectedLabel).lowercased()
        let domain = report.detectedDomain.lowercased()
        let sizeText = "Single grayscale sonographic frame (\(Int(imageSize.width)) x \(Int(imageSize.height)))."

        if domain == "fetal" {
            if result.contains("femur") {
                return "\(sizeText) Focused obstetric frame dominated by a single elongated echogenic long-bone structure with limited surrounding anatomy in view."
            }
            if result.contains("brain") || result.contains("head") {
                return "\(sizeText) Focused obstetric frame centered on the fetal head or intracranial region rather than a whole-fetus overview."
            }
            if result.contains("abdomen") {
                return "\(sizeText) Focused obstetric frame centered on the fetal abdomen or torso rather than a whole-fetus overview."
            }
            if result.contains("cervix") {
                return "\(sizeText) Focused obstetric cervical frame with limited surrounding fetal anatomy shown."
            }
        }

        if domain == "kidney" {
            return "\(sizeText) Focused renal ultrasound frame with the field of view concentrated on renal morphology rather than a broad abdominal survey."
        }

        if domain == "thyroid" {
            return "\(sizeText) Focused thyroid ultrasound frame centered on glandular soft tissue."
        }

        if domain == "carotid" {
            return "\(sizeText) Focused vascular ultrasound frame with vessel lumen and wall interface prioritized in view."
        }

        return "\(sizeText) Limited image-only description available from the submitted single frame."
    }

    private static func isFetalBiometryCase(_ report: PULSEAnalysisReport) -> Bool {
        report.detectedDomain.lowercased() == "fetal" && (
            measurementFinding(report: report) != nil
            || report.subviewText != nil
            || report.detectedLabel.lowercased().contains("femur")
            || report.detectedLabel.lowercased().contains("brain")
            || report.detectedLabel.lowercased().contains("abdomen")
            || report.detectedLabel.lowercased().contains("cervix")
        )
    }

    private static func primaryResult(_ report: PULSEAnalysisReport) -> String? {
        report.subviewText ?? report.detectedLabel
    }

    private static func measurementFinding(report: PULSEAnalysisReport) -> PULSEFinding? {
        report.specialistFindings.first { $0.category == .measurement }
    }

    private static func measurementLabel(report: PULSEAnalysisReport) -> String? {
        guard let label = measurementFinding(report: report)?.label?.pulseDisplayText, !label.isEmpty else {
            return nil
        }
        return label.uppercased()
    }

    static func measurementSummaryText(report: PULSEAnalysisReport) -> String? {
        guard let finding = measurementFinding(report: report) else { return nil }
        let label = finding.label?.pulseDisplayText.uppercased() ?? "MEASUREMENT"
        let detailText = formattedDetailText(from: finding.displayDetails, limit: 3)
        if detailText.isEmpty == false {
            return "Automated \(label) measurement was obtained on the submitted image (\(detailText))."
        }
        return "Automated \(label) measurement was obtained on the submitted image."
    }

    static func segmentationSummaryText(report: PULSEAnalysisReport) -> String? {
        let findings = report.specialistFindings.filter { $0.category == .segmentation }
        guard findings.isEmpty == false else { return nil }

        let summaries = findings.compactMap { finding -> String? in
            let detailText = formattedSegmentationDetailText(from: finding.displayDetails)
            if detailText.isEmpty == false {
                return "\(finding.title) output was generated (\(detailText))."
            }
            return "\(finding.title) output was generated on the submitted image."
        }

        guard summaries.isEmpty == false else { return nil }
        return summaries.joined(separator: " ")
    }

    private static func segmentationImpressionText(report: PULSEAnalysisReport) -> String? {
        let findings = report.specialistFindings.filter { $0.category == .segmentation }
        guard let first = findings.first else { return nil }
        let detailText = formattedSegmentationDetailText(from: first.displayDetails)
        if detailText.isEmpty == false {
            return "Segmentation-derived extent on the submitted frame: \(detailText)."
        }
        return "Segmentation output was generated on the submitted frame."
    }

    static func selectionRationaleText(report: PULSEAnalysisReport) -> String? {
        let primary = report.primaryFinding
        let routing = report.routingFinding
        let primaryLabel = primary?.label?.pulseDisplayText ?? primaryResult(report)

        guard primary != nil || routing != nil || primaryLabel != nil else {
            return nil
        }

        var sentences: [String] = []

        if let primary, let primaryLabel {
            let lowerLabel = primaryLabel.lowercased()
            switch primary.category {
            case .classification:
                if report.subviewText != nil {
                    var sentence = "Selection of \(lowerLabel) is driven by the highest-confidence subview or structure classifier"
                    if let confidence = primary.confidencePercentText {
                        sentence += " at \(confidence)"
                    }
                    sentence += "."
                    sentences.append(sentence)
                } else {
                    var sentence = "Selection of \(lowerLabel) is driven by the dedicated pathology classifier"
                    if let confidence = primary.confidencePercentText {
                        sentence += " at \(confidence)"
                    }
                    sentence += "."
                    sentences.append(sentence)
                }
            case .measurement:
                var sentence = "Selection of \(lowerLabel) is supported by successful measurement generation"
                if let confidence = primary.confidencePercentText {
                    sentence += " at \(confidence)"
                }
                sentence += "."
                sentences.append(sentence)
            case .segmentation:
                var sentence = "Selection of \(lowerLabel) is supported by a dedicated segmentation output"
                if let confidence = primary.confidencePercentText {
                    sentence += " at \(confidence)"
                }
                sentence += "."
                sentences.append(sentence)
            default:
                break
            }

            let summary = primary.summary.trimmingCharacters(in: .whitespacesAndNewlines)
            if summary.isEmpty == false {
                sentences.append("Classifier evidence: \(summary)")
            }
        }

        if let routing, let confidence = routing.confidencePercentText {
            sentences.append("Domain routing supports the \(report.detectedDomain.pulseDisplayText.lowercased()) domain at \(confidence).")
        }

        if let corroboration = supportingCorroborationText(report: report) {
            sentences.append(corroboration)
        }

        guard sentences.isEmpty == false else { return nil }
        return sentences.joined(separator: " ")
    }

    private static func impressionRationaleLine(report: PULSEAnalysisReport) -> String? {
        var clauses: [String] = []

        if let routing = report.routingFinding?.confidencePercentText {
            clauses.append("domain routing \(routing)")
        }

        if let primary = report.primaryFinding?.confidencePercentText {
            clauses.append("primary specialist confidence \(primary)")
        }

        if measurementFinding(report: report) != nil {
            clauses.append("successful measurement extraction")
        }

        if report.specialistFindings.contains(where: { $0.category == .segmentation }) {
            clauses.append("segmentation support")
        }

        guard clauses.isEmpty == false else { return nil }
        return "This interpretation is based on \(joinedList(clauses))."
    }

    private static func supportingCorroborationText(report: PULSEAnalysisReport) -> String? {
        var items: [String] = []

        if let label = measurementLabel(report: report) {
            items.append("\(label) measurement was successfully generated")
        }

        if let segmentation = report.specialistFindings.first(where: { $0.category == .segmentation }) {
            let detailText = formattedSegmentationDetailText(from: segmentation.displayDetails)
            if detailText.isEmpty == false {
                items.append("segmentation localized the finding (\(detailText))")
            } else {
                items.append("segmentation output was also generated")
            }
        }

        guard items.isEmpty == false else { return nil }
        return "Corroborating evidence includes \(joinedList(items))."
    }

    private static func pathologyFindings(in report: PULSEAnalysisReport) -> [PULSEFinding] {
        report.specialistFindings.filter { finding in
            let task = finding.taskID.lowercased()
            return task.contains("classification")
                && !task.contains("domain")
                && !task.contains("view")
                && !task.contains("subview")
                && !task.contains("organ")
                && !task.contains("plane")
        }
    }

    private static func primaryPathologyFinding(in report: PULSEAnalysisReport) -> PULSEFinding? {
        pathologyFindings(in: report).first
    }

    private static func formattedDetailText(from details: [String], limit: Int) -> String {
        details.prefix(limit)
            .map(formatDetailItem)
            .filter { !$0.isEmpty }
            .joined(separator: "; ")
    }

    private static func formattedSegmentationDetailText(from details: [String]) -> String {
        var pieces: [String] = []
        if let area = detailValue(for: "segmented_area_px", in: details) {
            pieces.append("segmented area \(area) px")
        }
        if let width = detailValue(for: "bbox_width_px", in: details),
           let height = detailValue(for: "bbox_height_px", in: details) {
            pieces.append("bounding box \(width) x \(height) px")
        }
        if let volume = detailValue(for: "volume", in: details) {
            pieces.append("volume \(volume)")
        }
        if pieces.isEmpty {
            return formattedDetailText(from: details, limit: 4)
        }
        return pieces.joined(separator: ", ")
    }

    private static func detailValue(for key: String, in details: [String]) -> String? {
        for detail in details {
            let parts = detail.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            if parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == key.lowercased() {
                return parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return nil
    }

    private static func formatDetailItem(_ detail: String) -> String {
        let parts = detail.split(separator: ":", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return detail }
        let rawKey = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
        let value = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
        let key = rawKey
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "px", with: "px")
        return "\(key): \(value)"
    }

    private static func numbered(_ lines: [String]) -> String {
        lines.enumerated()
            .map { "\($0.offset + 1). \($0.element)" }
            .joined(separator: "\n")
    }

    private static func joinedList(_ items: [String]) -> String {
        switch items.count {
        case 0:
            return ""
        case 1:
            return items[0]
        case 2:
            return "\(items[0]) and \(items[1])"
        default:
            return items.dropLast().joined(separator: ", ") + ", and " + (items.last ?? "")
        }
    }
}
