// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "yanaiengine",
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "yanaiengine",
            dependencies: [
                .product(name: "NIO", package: "swift-nio"),
            ],
            resources: [.process("gemm.metal")]
        ),
    ]
)
