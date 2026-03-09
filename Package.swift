// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "yanaiengine",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.0.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0-rc.1"),
    ],
    targets: [
        .executableTarget(
            name: "yanaiengine",
            dependencies: [
                .product(name: "NIO", package: "swift-nio"),
                .product(name: "Hummingbird", package: "hummingbird"),
            ],
            resources: [.process("gemm.metal")]
        ),
    ]
)
