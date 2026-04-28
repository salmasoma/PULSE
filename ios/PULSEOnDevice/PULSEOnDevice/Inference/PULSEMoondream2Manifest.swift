import Foundation

struct PULSEMoondream2Manifest: Decodable {
    let provider: String
    let variant: String?
    let modality: String?
    let textModelFilename: String
    let mmprojFilename: String
    let runtimeSupported: Bool?
    let runtimeTarget: String?
    let note: String?

    private enum CodingKeys: String, CodingKey {
        case provider
        case variant
        case modality
        case textModelFilename = "text_model_filename"
        case mmprojFilename = "mmproj_filename"
        case runtimeSupported = "runtime_supported"
        case runtimeTarget = "runtime_target"
        case note
    }

    var displayLabel: String {
        let providerText = provider.replacingOccurrences(of: "_", with: " ").capitalized
        if let variant, !variant.isEmpty {
            return "\(providerText) \(variant)"
        }
        return providerText
    }

    static func load() -> PULSEMoondream2Manifest? {
        let manifestNames = [
            "local_vqa_manifest",
            "moondream2_reasoning_manifest",
        ]

        for manifestName in manifestNames {
            guard
                let url = Bundle.main.findResource(
                    named: manifestName,
                    extension: "json",
                    preferredSubdirectories: ["Reasoning", "Resources/Reasoning"]
                ),
                let data = try? Data(contentsOf: url),
                let manifest = try? JSONDecoder().decode(PULSEMoondream2Manifest.self, from: data)
            else {
                continue
            }

            return manifest
        }

        return nil
    }
}
