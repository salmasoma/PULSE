// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "PULSEMoondreamVendor",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "PULSEMoondreamVendor", targets: ["PULSEMoondreamVendor"])
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "llama.xcframework"
        ),
        .target(
            name: "PULSEMoondreamVendor",
            dependencies: ["llama"],
            path: "Sources/PULSEMoondreamVendor"
        )
    ]
)
