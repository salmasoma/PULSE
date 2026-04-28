import Foundation
import UIKit

struct PULSEArtifactImage: Identifiable, Codable, Hashable {
    let id: UUID
    let kind: String
    let caption: String
    let pngData: Data

    init(id: UUID = UUID(), kind: String, caption: String, pngData: Data) {
        self.id = id
        self.kind = kind
        self.caption = caption
        self.pngData = pngData
    }
}

struct PULSEFinding: Identifiable, Codable, Hashable {
    let id: UUID
    let taskID: String
    let title: String
    let summary: String
    let label: String?
    let confidence: Double?
    let details: [String]
    let artifacts: [PULSEArtifactImage]

    init(
        id: UUID = UUID(),
        taskID: String,
        title: String,
        summary: String,
        label: String? = nil,
        confidence: Double? = nil,
        details: [String],
        artifacts: [PULSEArtifactImage] = []
    ) {
        self.id = id
        self.taskID = taskID
        self.title = title
        self.summary = summary
        self.label = label
        self.confidence = confidence
        self.details = details
        self.artifacts = artifacts
    }
}

struct PULSEQualitySummary: Codable, Hashable {
    let brightness: Double
    let contrast: Double
    let qualityLabel: String
}

struct PULSEStageTiming: Codable, Hashable {
    let stageID: String
    let title: String
    let elapsedMs: Double
    let count: Int?

    init(stageID: String, title: String, elapsedMs: Double, count: Int? = nil) {
        self.stageID = stageID
        self.title = title
        self.elapsedMs = elapsedMs
        self.count = count
    }
}

struct PULSEProfilingSummary: Codable, Hashable {
    let deviceIdentifier: String
    let perceptionMs: Double?
    let reasoningMs: Double?
    let endToEndMs: Double?
    let stages: [PULSEStageTiming]

    init(
        deviceIdentifier: String,
        perceptionMs: Double? = nil,
        reasoningMs: Double? = nil,
        endToEndMs: Double? = nil,
        stages: [PULSEStageTiming] = []
    ) {
        self.deviceIdentifier = deviceIdentifier
        self.perceptionMs = perceptionMs
        self.reasoningMs = reasoningMs
        self.endToEndMs = endToEndMs
        self.stages = stages
    }

    func withReasoningDuration(_ elapsedMs: Double) -> PULSEProfilingSummary {
        let updatedEndToEnd = (perceptionMs ?? 0) + elapsedMs
        let filteredStages = stages.filter { $0.stageID != "reasoning" }
        return PULSEProfilingSummary(
            deviceIdentifier: deviceIdentifier,
            perceptionMs: perceptionMs,
            reasoningMs: elapsedMs,
            endToEndMs: updatedEndToEnd,
            stages: filteredStages + [
                PULSEStageTiming(stageID: "reasoning", title: "Reasoned report generation", elapsedMs: elapsedMs)
            ]
        )
    }
}

struct PULSEReasonedReport: Codable, Hashable {
    let model: String
    let mode: String
    let output: String
    let generatedAt: Date

    init(model: String, mode: String = "reasoning", output: String, generatedAt: Date = Date()) {
        self.model = model
        self.mode = mode
        self.output = output
        self.generatedAt = generatedAt
    }
}

struct PULSEAnalysisReport: Identifiable, Codable, Hashable {
    let id: UUID
    let createdAt: Date
    let detectedDomain: String
    let detectedLabel: String
    let quality: PULSEQualitySummary
    let findings: [PULSEFinding]
    let note: String
    let reasonedReport: PULSEReasonedReport?
    let reasoningStatusMessage: String?
    let profiling: PULSEProfilingSummary?

    init(
        id: UUID = UUID(),
        createdAt: Date = Date(),
        detectedDomain: String,
        detectedLabel: String,
        quality: PULSEQualitySummary,
        findings: [PULSEFinding],
        note: String,
        reasonedReport: PULSEReasonedReport? = nil,
        reasoningStatusMessage: String? = nil,
        profiling: PULSEProfilingSummary? = nil
    ) {
        self.id = id
        self.createdAt = createdAt
        self.detectedDomain = detectedDomain
        self.detectedLabel = detectedLabel
        self.quality = quality
        self.findings = findings
        self.note = note
        self.reasonedReport = reasonedReport
        self.reasoningStatusMessage = reasoningStatusMessage
        self.profiling = profiling
    }

    func withReasonedReport(_ reasonedReport: PULSEReasonedReport?) -> PULSEAnalysisReport {
        PULSEAnalysisReport(
            id: id,
            createdAt: createdAt,
            detectedDomain: detectedDomain,
            detectedLabel: detectedLabel,
            quality: quality,
            findings: findings,
            note: note,
            reasonedReport: reasonedReport,
            reasoningStatusMessage: reasonedReport == nil ? reasoningStatusMessage : nil,
            profiling: profiling
        )
    }

    func withReasoningStatusMessage(_ message: String?) -> PULSEAnalysisReport {
        PULSEAnalysisReport(
            id: id,
            createdAt: createdAt,
            detectedDomain: detectedDomain,
            detectedLabel: detectedLabel,
            quality: quality,
            findings: findings,
            note: note,
            reasonedReport: reasonedReport,
            reasoningStatusMessage: message,
            profiling: profiling
        )
    }

    func withProfiling(_ profiling: PULSEProfilingSummary?) -> PULSEAnalysisReport {
        PULSEAnalysisReport(
            id: id,
            createdAt: createdAt,
            detectedDomain: detectedDomain,
            detectedLabel: detectedLabel,
            quality: quality,
            findings: findings,
            note: note,
            reasonedReport: reasonedReport,
            reasoningStatusMessage: reasoningStatusMessage,
            profiling: profiling
        )
    }
}

struct PULSESavedAnalysis: Identifiable, Codable, Hashable {
    let id: UUID
    let createdAt: Date
    let prompt: String
    let sourceName: String?
    let imageFilename: String
    let report: PULSEAnalysisReport

    var title: String {
        report.detectedLabel.pulseDisplayText
    }

    var subtitle: String {
        report.detectedDomain.pulseDisplayText
    }

    var displaySourceName: String {
        sourceName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? sourceName!
            : "Ultrasound Study"
    }

    func withReport(_ updatedReport: PULSEAnalysisReport) -> PULSESavedAnalysis {
        PULSESavedAnalysis(
            id: id,
            createdAt: createdAt,
            prompt: prompt,
            sourceName: sourceName,
            imageFilename: imageFilename,
            report: updatedReport
        )
    }
}

struct PULSEBatchPresentation: Identifiable, Hashable {
    let id: UUID
    let createdAt: Date
    let title: String
    let analyses: [PULSESavedAnalysis]

    init(id: UUID = UUID(), createdAt: Date = Date(), title: String, analyses: [PULSESavedAnalysis]) {
        self.id = id
        self.createdAt = createdAt
        self.title = title
        self.analyses = analyses
    }
}

struct PULSEIntakeStudy: Identifiable, Hashable {
    enum Source: String, Hashable {
        case camera
        case photoLibrary
        case files

        var title: String {
            switch self {
            case .camera:
                return "Camera"
            case .photoLibrary:
                return "Photo Library"
            case .files:
                return "Files"
            }
        }

        var symbol: String {
            switch self {
            case .camera:
                return "camera.fill"
            case .photoLibrary:
                return "photo.stack.fill"
            case .files:
                return "folder.fill"
            }
        }
    }

    let id: UUID
    let source: Source
    let sourceName: String
    let imageData: Data
    let createdAt: Date

    init(id: UUID = UUID(), source: Source, sourceName: String, imageData: Data, createdAt: Date = Date()) {
        self.id = id
        self.source = source
        self.sourceName = sourceName
        self.imageData = imageData
        self.createdAt = createdAt
    }

    var image: UIImage? {
        UIImage(data: imageData)
    }
}
