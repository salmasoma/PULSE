import CoreML
import Foundation

actor PULSEModelRegistry {
    static let shared = PULSEModelRegistry()

    private let manifest: PULSECoreMLManifest
    private var cache: [String: MLModel] = [:]

    init() {
        do {
            manifest = try PULSECoreMLManifest.loadFromBundle()
        } catch {
            manifest = PULSECoreMLManifest(
                name: "Missing manifest",
                sourceModelRoot: "",
                modelCount: 0,
                exportedCount: 0,
                skippedCount: 0,
                models: []
            )
        }
    }

    func currentManifest() -> PULSECoreMLManifest {
        manifest
    }

    func modelDescriptor(taskID: String) -> PULSECoreMLModelDescriptor? {
        manifest.models.first(where: { $0.taskID == taskID })
    }

    func descriptors(for domain: String) -> [PULSECoreMLModelDescriptor] {
        manifest.models.filter { $0.domain == domain && $0.taskID != "system/domain_classification" }
    }

    func loadModel(for descriptor: PULSECoreMLModelDescriptor) throws -> MLModel {
        if let cached = cache[descriptor.taskID] {
            return cached
        }
        let baseName = descriptor.outputName
        let resourceURL =
            Bundle.main.findResource(named: baseName, extension: "mlmodelc", preferredSubdirectories: ["Models", "Resources/Models"]) ??
            Bundle.main.findResource(named: baseName, extension: "mlpackage", preferredSubdirectories: ["Models", "Resources/Models"])
        guard let resourceURL else {
            throw PULSEOnDeviceError.missingModel(baseName)
        }
        let model = try MLModel(contentsOf: resourceURL)
        cache[descriptor.taskID] = model
        return model
    }
}
