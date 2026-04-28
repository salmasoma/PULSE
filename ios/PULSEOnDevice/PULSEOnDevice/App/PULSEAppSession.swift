import SwiftUI
import UIKit

@MainActor
final class PULSEAppSession: ObservableObject {
    enum Tab: Hashable {
        case home
        case history
    }

    @Published var hasCompletedSplash = false
    @Published var selectedTab: Tab = .home
    @Published var presentedAnalysis: PULSESavedAnalysis?
    @Published var presentedBatch: PULSEBatchPresentation?
    @Published private(set) var history: [PULSESavedAnalysis]
    @Published private(set) var manifestSummary: ManifestSummary

    let pipeline = PULSEOnDevicePipeline()

    private let historyStore = PULSEHistoryStore()

    init() {
        history = historyStore.loadHistory()
        manifestSummary = ManifestSummary.load()
    }

    func completeSplash() {
        withAnimation(.spring(response: 0.9, dampingFraction: 0.92)) {
            hasCompletedSplash = true
        }
    }

    func saveAnalysis(report: PULSEAnalysisReport, sourceImage: UIImage, prompt: String, sourceName: String? = nil) throws -> PULSESavedAnalysis {
        let saved = try historyStore.store(report: report, sourceImage: sourceImage, prompt: prompt, sourceName: sourceName)
        history = historyStore.loadHistory()
        return saved
    }

    func updateAnalysis(_ analysis: PULSESavedAnalysis) throws {
        try historyStore.update(analysis)
        history = historyStore.loadHistory()

        if presentedAnalysis?.id == analysis.id {
            presentedAnalysis = history.first(where: { $0.id == analysis.id }) ?? analysis
        }

        if let batch = presentedBatch {
            let updatedAnalyses = batch.analyses.map { existing in
                existing.id == analysis.id ? (history.first(where: { $0.id == analysis.id }) ?? analysis) : existing
            }
            presentedBatch = PULSEBatchPresentation(
                id: batch.id,
                createdAt: batch.createdAt,
                title: batch.title,
                analyses: updatedAnalyses
            )
        }
    }

    func recordAnalysis(report: PULSEAnalysisReport, sourceImage: UIImage, prompt: String, sourceName: String? = nil) throws {
        let saved = try saveAnalysis(report: report, sourceImage: sourceImage, prompt: prompt, sourceName: sourceName)
        presentedAnalysis = saved
    }

    func openHistory(_ analysis: PULSESavedAnalysis) {
        presentedBatch = nil
        presentedAnalysis = analysis
    }

    func presentBatch(_ analyses: [PULSESavedAnalysis]) {
        guard analyses.isEmpty == false else { return }
        presentedAnalysis = nil
        presentedBatch = PULSEBatchPresentation(
            title: analyses.count == 1 ? "Study Result" : "Batch Results",
            analyses: analyses
        )
    }

    func image(for analysis: PULSESavedAnalysis) -> UIImage? {
        historyStore.loadImage(for: analysis)
    }
}

struct ManifestSummary {
    let exportedCount: Int
    let variantLabel: String

    static func load() -> ManifestSummary {
        let specialistManifest = try? PULSEFetalSpecialistManifest.loadFromBundle()
        let specialistCount = (specialistManifest?.mobileFetalClip != nil ? 1 : 0) + (specialistManifest?.fetalnet != nil ? 1 : 0)
        do {
            let manifest = try PULSECoreMLManifest.loadFromBundle()
            let variants = Set(manifest.models.map(\.modelVariant))
            return ManifestSummary(
                exportedCount: manifest.exportedCount + specialistCount,
                variantLabel: variants.count == 1 ? (variants.first ?? "unknown") : "mixed"
            )
        } catch {
            if specialistCount > 0 {
                return ManifestSummary(exportedCount: specialistCount, variantLabel: "mixed")
            }
            return .placeholder
        }
    }

    static let placeholder = ManifestSummary(exportedCount: 0, variantLabel: "none")
}
