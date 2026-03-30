import AppKit
import SwiftUI

enum AppTheme {
    static let accent = Color(nsColor: .controlAccentColor)
    static let destructive = Color(nsColor: .systemRed)
    static let success = Color(nsColor: .systemGreen)
    static let warning = Color(nsColor: .systemOrange)

    static let secondaryText = Color(nsColor: .secondaryLabelColor)
    static let tertiaryText = Color(nsColor: .tertiaryLabelColor)
    static let separator = Color(nsColor: .separatorColor).opacity(0.55)
    static let strongSeparator = Color(nsColor: .separatorColor)
    static let panelFill = Color(nsColor: .controlBackgroundColor).opacity(0.42)
    static let badgeFill = Color(nsColor: .controlBackgroundColor).opacity(0.64)
    static let inputFill = Color(nsColor: .controlBackgroundColor).opacity(0.92)
    static let actionCardFill = Color(nsColor: .labelColor).opacity(0.02)
    static let actionCardHoverFill = Color(nsColor: .labelColor).opacity(0.08)

    static let background = LinearGradient(
        colors: [
            Color(nsColor: .windowBackgroundColor),
            Color(nsColor: .underPageBackgroundColor).opacity(0.92)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let upperGlow = RadialGradient(
        colors: [
            accent.opacity(0.10),
            accent.opacity(0.03),
            .clear
        ],
        center: .topTrailing,
        startRadius: 16,
        endRadius: 320
    )

    static let lowerGlow = RadialGradient(
        colors: [
            Color.white.opacity(0.05),
            .clear
        ],
        center: .bottomLeading,
        startRadius: 20,
        endRadius: 280
    )
}

struct AppBackdrop: View {
    var body: some View {
        ZStack {
            AppTheme.background
            AppTheme.upperGlow
            AppTheme.lowerGlow
        }
        .ignoresSafeArea()
    }
}

private struct SurfaceModifier: ViewModifier {
    let cornerRadius: CGFloat
    let material: Material
    let shadowOpacity: Double

    func body(content: Content) -> some View {
        content
            .background(material, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(AppTheme.separator, lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(shadowOpacity), radius: 14, y: 8)
    }
}

private struct ShellSurfaceModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .strokeBorder(AppTheme.separator, lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.14), radius: 24, y: 14)
    }
}

private struct FrostedCardModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(16)
            .modifier(SurfaceModifier(cornerRadius: 20, material: .thinMaterial, shadowOpacity: 0.03))
    }
}

extension View {
    func shellSurface() -> some View {
        modifier(ShellSurfaceModifier())
    }

    func sectionSurface(cornerRadius: CGFloat = 22, material: Material = .thinMaterial, shadowOpacity: Double = 0.08) -> some View {
        modifier(SurfaceModifier(cornerRadius: cornerRadius, material: material, shadowOpacity: shadowOpacity))
    }

    func frostedCard() -> some View {
        modifier(FrostedCardModifier())
    }
}
