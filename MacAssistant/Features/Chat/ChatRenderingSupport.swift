import AppKit
import Foundation
import ImageIO
import SwiftUI

private final class CachedNSImageBox: NSObject {
    let image: NSImage

    init(image: NSImage) {
        self.image = image
    }
}

private final class CachedAttributedStringBox: NSObject {
    let value: AttributedString?

    init(value: AttributedString?) {
        self.value = value
    }
}

struct AttachmentImageRequest: Hashable {
    let url: URL
    let maxPixelSize: Int

    var cacheKey: NSString {
        "\(url.path)#\(maxPixelSize)" as NSString
    }

    var taskID: String {
        cacheKey as String
    }
}

enum AttachmentImageCache {
    nonisolated(unsafe) private static let cache = NSCache<NSString, CachedNSImageBox>()

    static func cachedImage(for request: AttachmentImageRequest) -> NSImage? {
        cache.object(forKey: request.cacheKey)?.image
    }

    @MainActor
    static func loadImage(for request: AttachmentImageRequest) async -> NSImage? {
        if let cachedImage = cachedImage(for: request) {
            return cachedImage
        }

        return await withCheckedContinuation { continuation in
            let cacheKey = request.taskID
            DispatchQueue.global(qos: .userInitiated).async {
                let image = downsampledImage(at: request.url, maxPixelSize: request.maxPixelSize)
                    ?? NSImage(contentsOf: request.url)
                if let image {
                    cache.setObject(CachedNSImageBox(image: image), forKey: cacheKey as NSString)
                }
                DispatchQueue.main.async {
                    continuation.resume(returning: image)
                }
            }
        }
    }

    private static func downsampledImage(at url: URL, maxPixelSize: Int) -> NSImage? {
        guard maxPixelSize > 0 else { return nil }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
            kCGImageSourceShouldCacheImmediately: false,
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}

enum MarkdownTextCache {
    nonisolated(unsafe) private static let cache = NSCache<NSString, CachedAttributedStringBox>()

    static func attributedString(for text: String) -> AttributedString? {
        let cacheKey = text as NSString
        if let cached = cache.object(forKey: cacheKey) {
            return cached.value
        }

        let parsed = try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .full,
                failurePolicy: .returnPartiallyParsedIfPossible
            )
        )
        cache.setObject(CachedAttributedStringBox(value: parsed), forKey: cacheKey)
        return parsed
    }
}

struct CachedAttachmentImage<Placeholder: View>: View {
    let request: AttachmentImageRequest
    let contentMode: ContentMode
    let imagePadding: CGFloat
    private let placeholder: Placeholder

    @State private var image: NSImage?

    init(
        request: AttachmentImageRequest,
        contentMode: ContentMode,
        imagePadding: CGFloat = 0,
        @ViewBuilder placeholder: () -> Placeholder
    ) {
        self.request = request
        self.contentMode = contentMode
        self.imagePadding = imagePadding
        self.placeholder = placeholder()
    }

    var body: some View {
        Group {
            if let image {
                if contentMode == .fill {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .padding(imagePadding)
                }
            } else {
                placeholder
            }
        }
        .task(id: request.taskID) {
            image = AttachmentImageCache.cachedImage(for: request)
            if image == nil {
                image = await AttachmentImageCache.loadImage(for: request)
            }
        }
    }
}

struct CachedMarkdownText: View {
    let text: String
    private let parsedMarkdown: AttributedString?

    init(text: String) {
        self.text = text
        self.parsedMarkdown = MarkdownTextCache.attributedString(for: text)
    }

    var body: some View {
        if let parsedMarkdown {
            Text(parsedMarkdown)
        } else {
            Text(text)
        }
    }
}
