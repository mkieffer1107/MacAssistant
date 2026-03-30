import SwiftUI

enum SettingsTab: String, CaseIterable, Hashable {
    case general
    case modelFiles
    case runtimeLogs

    var title: String {
        switch self {
        case .general:
            return "General"
        case .modelFiles:
            return "Model Files"
        case .runtimeLogs:
            return "Runtime Logs"
        }
    }

    var systemImage: String {
        switch self {
        case .general:
            return "gearshape"
        case .modelFiles:
            return "externaldrive"
        case .runtimeLogs:
            return "terminal"
        }
    }
}

struct SettingsView: View {
    let model: AppModel
    @State private var selectedTab: SettingsTab = .general

    private var defaultVoiceSelection: Binding<String> {
        Binding(
            get: { model.settings.defaultVoicePreset },
            set: { model.settings.defaultVoicePreset = $0 }
        )
    }

    private var inputDeviceSelection: Binding<String> {
        Binding(
            get: { model.selectedInputDevicePickerValue },
            set: { model.setSelectedInputDevicePickerValue($0) }
        )
    }

    private var launchOnOpenSelection: Binding<Bool> {
        Binding(
            get: { model.settings.launchOnOpen },
            set: { model.settings.launchOnOpen = $0 }
        )
    }

    private var alwaysAcceptToolCallsSelection: Binding<Bool> {
        Binding(
            get: { model.settings.alwaysAcceptToolCalls },
            set: { model.settings.alwaysAcceptToolCalls = $0 }
        )
    }

    private var streamReplySpeechSelection: Binding<Bool> {
        Binding(
            get: { model.settings.streamReplySpeechWhileGenerating },
            set: { model.settings.streamReplySpeechWhileGenerating = $0 }
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            tabSelector
            selectedPane
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.clear)
        .onAppear {
            selectedTab = .general
            noteSelectedTab()
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Settings")
                    .font(.largeTitle.weight(.semibold))
                Text("Voice, launch behavior, runtime visibility, and permissions.")
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
            }

            Spacer(minLength: 0)

            Button("Done", action: closeSettings)
                .buttonStyle(.borderedProminent)
                .tint(AppTheme.accent)
                .controlSize(.large)
        }
        .padding(.horizontal, 18)
        .padding(.top, 16)
        .padding(.bottom, 6)
    }

    private var tabSelector: some View {
        HStack(spacing: 10) {
            ForEach(SettingsTab.allCases, id: \.self) { tab in
                SettingsTabButton(
                    tab: tab,
                    isSelected: selectedTab == tab,
                    action: { selectTab(tab) }
                )
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 10)
    }

    private var selectedPane: some View {
        Group {
            switch selectedTab {
            case .general:
                SettingsGeneralPane(
                    snapshot: model.settingsGeneralSnapshot,
                    defaultVoiceSelection: defaultVoiceSelection,
                    inputDeviceSelection: inputDeviceSelection,
                    launchOnOpenSelection: launchOnOpenSelection,
                    alwaysAcceptToolCallsSelection: alwaysAcceptToolCallsSelection,
                    streamReplySpeechSelection: streamReplySpeechSelection
                )
            case .modelFiles:
                let snapshot = model.settingsModelFilesSnapshot
                SettingsModelFilesPane(
                    entries: snapshot.entries,
                    openModelsFolder: model.openModelsFolderInFinder,
                    onAction: model.handleSettingsModelFileAction
                )
            case .runtimeLogs:
                SettingsRuntimeLogsPane(
                    snapshot: model.settingsRuntimeLogsSnapshot,
                    copyRuntimeLogs: model.copyRuntimeLogs
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.horizontal, 18)
        .padding(.top, 10)
        .padding(.bottom, 18)
    }

    private func selectTab(_ tab: SettingsTab) {
        guard selectedTab != tab else { return }
        selectedTab = tab
        noteSelectedTab()
    }

    private func noteSelectedTab() {
        model.appendSettingsNavigationDiagnostic(
            tabName: selectedTab.title,
            modelFileRowCount: model.settingsModelFilesSnapshot.entries.count
        )
    }

    private func closeSettings() {
        model.showingSettings = false
    }
}

private struct SettingsTabButton: View {
    let tab: SettingsTab
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(tab.title, systemImage: tab.systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(isSelected ? Color.white : Color.primary)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    Capsule(style: .continuous)
                        .fill(isSelected ? AppTheme.accent : AppTheme.actionCardFill)
                )
        }
        .buttonStyle(.plain)
    }
}
