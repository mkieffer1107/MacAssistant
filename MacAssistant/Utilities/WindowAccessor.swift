import AppKit
import SwiftUI

struct WindowAccessor: NSViewRepresentable {
    final class AccessorView: NSView {
        var onResolve: ((NSWindow) -> Void)?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let window {
                DispatchQueue.main.async { [weak self, weak window] in
                    guard let self, let window else { return }
                    self.onResolve?(window)
                }
            }
        }
    }

    var onResolve: (NSWindow) -> Void

    func makeNSView(context: Context) -> AccessorView {
        let view = AccessorView()
        view.onResolve = onResolve
        return view
    }

    func updateNSView(_ nsView: AccessorView, context: Context) {
        nsView.onResolve = onResolve
    }
}

enum FloatingAssistantWindowConfiguration {
    static let defaultContentSize = NSSize(width: 760, height: 640)
    static let minimumContentSize = NSSize(width: 640, height: 560)

    @MainActor
    static func apply(to window: NSWindow) {
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.titlebarSeparatorStyle = .none
        window.toolbarStyle = .unifiedCompact
        window.isOpaque = false
        window.backgroundColor = .clear
        window.isMovableByWindowBackground = true
        window.standardWindowButton(.zoomButton)?.isHidden = true

        // Join every Space and other apps' full-screen spaces like a floating overlay.
        var collectionBehavior = window.collectionBehavior
        collectionBehavior.remove(.moveToActiveSpace)
        collectionBehavior.remove(.fullScreenPrimary)
        collectionBehavior.remove(.fullScreenNone)
        collectionBehavior.insert(.fullScreenAuxiliary)
        collectionBehavior.insert(.canJoinAllSpaces)
        collectionBehavior.insert(.canJoinAllApplications)
        window.collectionBehavior = collectionBehavior

        window.setContentSize(defaultContentSize)
        window.minSize = minimumContentSize
    }
}
