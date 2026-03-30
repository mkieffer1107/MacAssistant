import SwiftUI
import UniformTypeIdentifiers

struct RootView: View {
    @Bindable var model: AppModel
    @State private var isImageDropTargeted = false

    var body: some View {
        ZStack {
            AppBackdrop()

            Group {
                switch model.phase {
                case .bootstrapping, .loading:
                    LoadingView(model: model)
                case .setup:
                    SetupView(model: model)
                case .ready:
                    if model.showingSettings {
                        SettingsView(model: model)
                    } else {
                        ChatView(model: model)
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
            .shellSurface()
            .padding(10)
            .overlay {
                if isImageDropTargeted && isImageDropEnabled {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(AppTheme.accent.opacity(0.10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 28, style: .continuous)
                                .strokeBorder(AppTheme.accent.opacity(0.65), style: StrokeStyle(lineWidth: 2, dash: [8, 8]))
                        )
                        .padding(10)
                        .transition(.opacity)
                }
            }
        }
        .onDrop(
            of: [UTType.fileURL.identifier, UTType.image.identifier],
            isTargeted: imageDropTargetBinding
        ) { providers in
            guard isImageDropEnabled else { return false }
            return model.importDroppedImage(from: providers)
        }
        .onChange(of: model.showingSettings) { _, isShowingSettings in
            if isShowingSettings {
                isImageDropTargeted = false
            }
        }
        .background(
            WindowAccessor { window in
                FloatingAssistantWindowConfiguration.apply(to: window)
            }
        )
    }

    private var isImageDropEnabled: Bool {
        model.phase == .ready && !model.showingSettings
    }

    private var imageDropTargetBinding: Binding<Bool> {
        Binding(
            get: { isImageDropEnabled && isImageDropTargeted },
            set: { newValue in
                isImageDropTargeted = isImageDropEnabled ? newValue : false
            }
        )
    }
}
