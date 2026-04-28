import Foundation

struct PULSECoreMLManifest: Decodable {
    let name: String
    let sourceModelRoot: String
    let modelCount: Int
    let exportedCount: Int
    let skippedCount: Int
    let models: [PULSECoreMLModelDescriptor]

    enum CodingKeys: String, CodingKey {
        case name
        case sourceModelRoot = "source_model_root"
        case modelCount = "model_count"
        case exportedCount = "exported_count"
        case skippedCount = "skipped_count"
        case models
    }

    static func loadFromBundle() throws -> PULSECoreMLManifest {
        guard let url = Bundle.main.findResource(named: "pulse_coreml_manifest", extension: "json", preferredSubdirectories: ["Models", "Resources/Models"]) else {
            throw PULSEOnDeviceError.missingManifest
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PULSECoreMLManifest.self, from: data)
    }
}

struct PULSECoreMLModelDescriptor: Decodable, Identifiable, Hashable {
    let taskID: String
    let outputName: String
    let domain: String
    let title: String
    let taskType: String
    let labels: [String]
    let modalities: [String]
    let imageSize: Int
    let runtimeEnabled: Bool
    let outputSemantics: String
    let inputNames: [String]
    let outputNames: [String]
    let sourceCheckpoint: String
    let modelVariant: String
    let studentWidth: Int?
    let coreMLPath: String?
    let exported: Bool
    let reason: String

    var id: String { taskID }

    enum CodingKeys: String, CodingKey {
        case taskID = "task_id"
        case outputName = "output_name"
        case domain
        case title
        case taskType = "task_type"
        case labels
        case modalities
        case imageSize = "image_size"
        case runtimeEnabled = "runtime_enabled"
        case outputSemantics = "output_semantics"
        case inputNames = "input_names"
        case outputNames = "output_names"
        case sourceCheckpoint = "source_checkpoint"
        case modelVariant = "model_variant"
        case studentWidth = "student_width"
        case coreMLPath = "coreml_path"
        case exported
        case reason
    }
}

enum PULSEOnDeviceError: LocalizedError {
    case missingManifest
    case missingModel(String)
    case invalidImage
    case invalidPrediction(String)

    var errorDescription: String? {
        switch self {
        case .missingManifest:
            return "The Core ML export manifest is missing from the app bundle."
        case .missingModel(let name):
            return "The compiled Core ML model `\(name)` is missing from the app bundle."
        case .invalidImage:
            return "The selected image could not be converted into a model input."
        case .invalidPrediction(let message):
            return message
        }
    }
}

extension Bundle {
    func findResource(named name: String, extension ext: String, preferredSubdirectories: [String]) -> URL? {
        for subdirectory in preferredSubdirectories {
            if let url = url(forResource: name, withExtension: ext, subdirectory: subdirectory) {
                return url
            }
        }

        if let url = url(forResource: name, withExtension: ext) {
            return url
        }

        let filename = "\(name).\(ext)"
        let resourceRoot = resourceURL ?? bundleURL
        let directCandidates = preferredSubdirectories.map { resourceRoot.appendingPathComponent($0).appendingPathComponent(filename) }
        for candidate in directCandidates where FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }

        guard let enumerator = FileManager.default.enumerator(at: resourceRoot, includingPropertiesForKeys: nil) else {
            return nil
        }
        for case let url as URL in enumerator {
            if url.lastPathComponent == filename {
                return url
            }
        }
        return nil
    }
}
