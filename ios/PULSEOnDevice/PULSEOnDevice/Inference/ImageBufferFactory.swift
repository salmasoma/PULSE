import CoreGraphics
import CoreML
import Foundation
import UIKit

enum ImageBufferFactory {
    static func pixelBuffer(from image: UIImage, size: Int) throws -> CVPixelBuffer {
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32ARGB,
            attributes as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw PULSEOnDeviceError.invalidImage
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            throw PULSEOnDeviceError.invalidImage
        }

        context.clear(CGRect(x: 0, y: 0, width: size, height: size))
        context.interpolationQuality = .high

        guard let rendered = image.renderedSquare(to: size), let cgImage = rendered.cgImage else {
            throw PULSEOnDeviceError.invalidImage
        }
        // Match training/export preprocessing: direct square resize without letterboxing.
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        return pixelBuffer
    }

    static func grayscaleStatistics(for image: UIImage, sampleSize: Int = 128) -> (brightness: Double, contrast: Double)? {
        guard let resized = image.resizeSquare(to: sampleSize), let cgImage = resized.cgImage else {
            return nil
        }
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var data = [UInt8](repeating: 0, count: bytesPerRow * height)
        guard let context = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var values = [Double]()
        values.reserveCapacity(width * height)
        for index in stride(from: 0, to: data.count, by: 4) {
            let r = Double(data[index]) / 255.0
            let g = Double(data[index + 1]) / 255.0
            let b = Double(data[index + 2]) / 255.0
            values.append((0.299 * r) + (0.587 * g) + (0.114 * b))
        }
        guard !values.isEmpty else {
            return nil
        }
        let mean = values.reduce(0.0, +) / Double(values.count)
        let variance = values.reduce(0.0) { partial, value in
            partial + pow(value - mean, 2.0)
        } / Double(values.count)
        return (brightness: mean, contrast: sqrt(variance))
    }

    static func modelCanvas(from image: UIImage, size: Int) -> UIImage? {
        image.renderedSquare(to: size)
    }
}

private extension UIImage {
    func resizeSquare(to side: Int) -> UIImage? {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format)
        return renderer.image { _ in
            let rect = CGRect(x: 0, y: 0, width: side, height: side)
            self.draw(in: rect)
        }
    }

    func renderedSquare(to side: Int) -> UIImage? {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format)
        return renderer.image { _ in
            self.draw(in: CGRect(x: 0, y: 0, width: side, height: side))
        }
    }
}
