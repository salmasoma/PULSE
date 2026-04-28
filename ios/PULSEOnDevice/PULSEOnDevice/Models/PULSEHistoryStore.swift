import Foundation
import UIKit

final class PULSEHistoryStore {
    private let fileManager = FileManager.default
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    private let rootURL: URL
    private let indexURL: URL
    private let imagesURL: URL

    init() {
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        rootURL = documentsURL.appendingPathComponent("PULSEHistory", isDirectory: true)
        indexURL = rootURL.appendingPathComponent("history.json")
        imagesURL = rootURL.appendingPathComponent("Images", isDirectory: true)

        encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        createDirectoriesIfNeeded()
    }

    func loadHistory() -> [PULSESavedAnalysis] {
        guard let data = try? Data(contentsOf: indexURL),
              let items = try? decoder.decode([PULSESavedAnalysis].self, from: data)
        else {
            return []
        }
        return items.sorted { $0.createdAt > $1.createdAt }
    }

    func loadImage(for analysis: PULSESavedAnalysis) -> UIImage? {
        let url = imagesURL.appendingPathComponent(analysis.imageFilename)
        return UIImage(contentsOfFile: url.path)
    }

    func store(report: PULSEAnalysisReport, sourceImage: UIImage, prompt: String, sourceName: String? = nil) throws -> PULSESavedAnalysis {
        createDirectoriesIfNeeded()

        let imageFilename = "\(report.id.uuidString).jpg"
        let imageURL = imagesURL.appendingPathComponent(imageFilename)

        guard let data = sourceImage.jpegData(compressionQuality: 0.92) else {
            throw PULSEHistoryStoreError.imageEncodingFailed
        }
        try data.write(to: imageURL, options: .atomic)

        let record = PULSESavedAnalysis(
            id: report.id,
            createdAt: report.createdAt,
            prompt: prompt,
            sourceName: sourceName,
            imageFilename: imageFilename,
            report: report
        )

        var items = loadHistory()
        items.removeAll { $0.id == record.id }
        items.insert(record, at: 0)

        let encoded = try encoder.encode(items)
        try encoded.write(to: indexURL, options: .atomic)
        return record
    }

    func update(_ analysis: PULSESavedAnalysis) throws {
        createDirectoriesIfNeeded()
        var items = loadHistory()
        items.removeAll { $0.id == analysis.id }
        items.insert(analysis, at: 0)
        let encoded = try encoder.encode(items)
        try encoded.write(to: indexURL, options: .atomic)
    }

    private func createDirectoriesIfNeeded() {
        try? fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true, attributes: nil)
        try? fileManager.createDirectory(at: imagesURL, withIntermediateDirectories: true, attributes: nil)
    }
}

enum PULSEHistoryStoreError: LocalizedError {
    case imageEncodingFailed

    var errorDescription: String? {
        switch self {
        case .imageEncodingFailed:
            return "The source image could not be encoded for local history."
        }
    }
}
