import SwiftUI

struct SettingsModelFilesPane: View {
    let entries: [ModelFileEntry]
    let openModelsFolder: () -> Void
    let onAction: (ModelFileAction) -> Void

    var body: some View {
        ScrollView {
            SettingsSectionCard {
                SettingsSectionHeader(
                    systemImage: "externaldrive",
                    title: "Model Files",
                    subtitle: "Delete individual downloaded models or open the storage folder in Finder."
                ) {
                    Button("Open in Finder", action: openModelsFolder)
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                }

                VStack(spacing: 0) {
                    ForEach(Array(entries.enumerated()), id: \.element.id) { index, entry in
                        ModelFileRow(
                            title: entry.title,
                            summary: entry.summary,
                            systemImage: entry.systemImage,
                            installState: entry.installState,
                            warmState: entry.warmState,
                            lastError: entry.lastError,
                            progress: entry.progress,
                            actionTitle: entry.actionTitle,
                            action: entry.action,
                            onAction: onAction
                        )

                        if index < entries.count - 1 {
                            SettingsRowDivider()
                                .padding(.leading, 58)
                        }
                    }
                }
            }
            .padding(.vertical, 6)
        }
        .scrollIndicators(.hidden)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
