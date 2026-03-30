import AppKit
import SwiftUI

struct SettingsView: View {
    @Bindable var model: AppModel

    private let builtInVoicePresets: [(id: String, title: String)] = [
        ("casual_male", "Casual Male"),
        ("casual_female", "Casual Female"),
        ("cheerful_female", "Cheerful Female"),
        ("neutral_male", "Neutral Male"),
        ("neutral_female", "Neutral Female")
    ]

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

            Divider()
                .overlay(AppTheme.separator)

            ScrollView {
                VStack(spacing: 14) {
                    generalSection
                    modelFilesSection
                    runtimeLogsSection
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 16)
            }
            .scrollIndicators(.hidden)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.clear)
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
        .padding(.bottom, 14)
    }

    private var generalSection: some View {
        SettingsSectionCard {
            SettingsSectionHeader(
                systemImage: "slider.horizontal.3",
                title: "General",
                subtitle: "Voice defaults and automation behavior."
            )

            SettingsRow(
                title: "Input Device",
                subtitle: "Automatic prefers the built-in microphone when one is available.",
                controlWidth: 320
            ) {
                inputDevicePicker
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Default Voice",
                subtitle: "Choose the voice used for spoken replies.",
                controlWidth: 220
            ) {
                defaultVoicePicker
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Launch on login",
                subtitle: "Open MacAssistant automatically after you sign in."
            ) {
                Toggle("", isOn: launchOnOpenSelection)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Always accept tool calls",
                subtitle: "Skip approval prompts when the runtime proposes a tool call."
            ) {
                Toggle("", isOn: alwaysAcceptToolCallsSelection)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }

            SettingsRowDivider()

            SettingsRow(
                title: "Stream spoken replies",
                subtitle: "Start speaking before the full reply finishes generating."
            ) {
                Toggle("", isOn: streamReplySpeechSelection)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }

    private var modelFilesSection: some View {
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
                ForEach(Array(modelFileEntries.enumerated()), id: \.element.id) { index, entry in
                    ModelFileRow(
                        title: entry.title,
                        summary: entry.summary,
                        systemImage: entry.systemImage,
                        installState: entry.installState,
                        warmState: entry.warmState,
                        lastError: entry.lastError,
                        progress: entry.progress,
                        actionTitle: entry.actionTitle,
                        action: entry.action
                    )

                    if index < modelFileEntries.count - 1 {
                        SettingsRowDivider()
                            .padding(.leading, 58)
                    }
                }
            }
        }
    }

    private var runtimeLogsSection: some View {
        SettingsSectionCard {
            SettingsSectionHeader(
                systemImage: "terminal",
                title: "Runtime Logs",
                subtitle: "Recent runtime and microphone diagnostics."
            ) {
                Button("Copy", action: copyRuntimeLogs)
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                    .disabled(model.runtimeLogsText.isEmpty)
            }

            ScrollView {
                Text(model.runtimeLogsText.isEmpty ? "No runtime logs yet." : model.runtimeLogsText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(model.runtimeLogsText.isEmpty ? AppTheme.secondaryText : Color.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(minHeight: 108, maxHeight: 144)
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(AppTheme.actionCardFill)
            )
        }
    }

    private var inputDevicePicker: some View {
        Picker("Input Device", selection: inputDeviceSelection) {
            Text("Automatic").tag("")
            ForEach(model.availableInputDevices) { device in
                Text(deviceLabel(for: device))
                    .tag(device.uid)
            }
            if let unavailableUID = model.unavailableSelectedInputDeviceUID {
                Text("Unavailable Device (\(unavailableUID))")
                    .tag(unavailableUID)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .controlSize(.regular)
    }

    private var defaultVoicePicker: some View {
        Picker("Default Voice", selection: defaultVoiceSelection) {
            ForEach(builtInVoicePresets, id: \.id) { voice in
                Text(voice.title)
                    .tag(voice.id)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .controlSize(.regular)
    }

    private func closeSettings() {
        model.showingSettings = false
    }

    private func openModelsFolder() {
        model.openModelsFolderInFinder()
    }

    private func copyRuntimeLogs() {
        model.copyRuntimeLogs()
    }

    private func handleModelAction(for installable: ModelInstallable) {
        switch installable.installState {
        case .installed:
            model.delete(installable)
        case .missing, .failed:
            model.install(installable)
        case .downloading:
            break
        }
    }

    private func modelActionTitle(for installable: ModelInstallable) -> String? {
        switch installable.installState {
        case .installed:
            return "Delete"
        case .missing:
            return "Install"
        case .failed:
            return "Retry"
        case .downloading:
            return nil
        }
    }

    private var modelFileEntries: [ModelFileEntry] {
        RuntimeModelID.allCases.map(modelFileEntry(for:))
    }

    private func modelFileEntry(for modelID: RuntimeModelID) -> ModelFileEntry {
        let installable = installable(for: modelID)
        let underlyingState = underlyingModelState(for: modelID)

        return ModelFileEntry(
            id: modelID.rawValue,
            title: modelFileTitle(for: modelID),
            summary: modelID.summary,
            systemImage: modelID.systemImage,
            installState: underlyingState.installState,
            warmState: underlyingState.warmState,
            lastError: underlyingState.lastError ?? installable?.lastError,
            progress: installable.flatMap { model.downloadProgress(for: $0.id) },
            actionTitle: installable.flatMap { modelActionTitle(for: $0, installState: underlyingState.installState) },
            action: { handleModelAction(for: modelID, installable: installable, installState: underlyingState.installState) }
        )
    }

    private func installable(for modelID: RuntimeModelID) -> ModelInstallable? {
        let installableID = switch modelID {
        case .agentModel:
            "agent_model"
        case .ttsModel, .sttModel:
            "voice_pack"
        }

        return model.modelInstallables.first(where: { $0.id == installableID })
    }

    private func underlyingModelState(for modelID: RuntimeModelID) -> UnderlyingModelState {
        model.underlyingModels[modelID.rawValue]
            ?? UnderlyingModelState(id: modelID.rawValue, installState: .missing, warmState: .cold, lastError: nil)
    }

    private func modelFileTitle(for modelID: RuntimeModelID) -> String {
        switch modelID {
        case .sttModel:
            return "Transcription"
        case .agentModel, .ttsModel:
            return modelID.title
        }
    }

    private func handleModelAction(for modelID: RuntimeModelID, installable: ModelInstallable?, installState: ModelInstallState) {
        switch installState {
        case .installed:
            model.deleteModel(modelID)
        case .missing, .failed:
            guard let installable else { return }
            handleModelAction(for: installable)
        case .downloading:
            break
        }
    }

    private func modelActionTitle(for installable: ModelInstallable, installState: ModelInstallState) -> String? {
        switch installState {
        case .installed:
            return "Delete"
        case .missing:
            return installable.installState == .downloading ? nil : "Install"
        case .failed:
            return installable.installState == .downloading ? nil : "Retry"
        case .downloading:
            return nil
        }
    }

    private func deviceLabel(for device: MicrophoneCaptureService.InputDevice) -> String {
        let suffix: String
        switch device.transport {
        case .builtIn:
            suffix = "Built-In"
        case .bluetooth:
            suffix = "Bluetooth"
        case .usb:
            suffix = "USB"
        case .aggregate:
            suffix = "Aggregate"
        case .virtual:
            suffix = "Virtual"
        case .unknown:
            suffix = "External"
        }
        return "\(device.name) (\(suffix))"
    }
}

private struct SettingsSectionCard<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            content
        }
        .padding(16)
        .sectionSurface(cornerRadius: 20, material: .thinMaterial, shadowOpacity: 0.03)
    }
}

private struct SettingsSectionHeader<Accessory: View>: View {
    let systemImage: String
    let title: String
    let subtitle: String
    private let accessory: Accessory

    init(
        systemImage: String,
        title: String,
        subtitle: String,
        @ViewBuilder accessory: () -> Accessory
    ) {
        self.systemImage = systemImage
        self.title = title
        self.subtitle = subtitle
        self.accessory = accessory()
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(AppTheme.secondaryText)
                .frame(width: 22, height: 22)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.title3.weight(.semibold))
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
            }

            Spacer(minLength: 0)

            accessory
        }
    }
}

private extension SettingsSectionHeader where Accessory == EmptyView {
    init(systemImage: String, title: String, subtitle: String) {
        self.init(systemImage: systemImage, title: title, subtitle: subtitle) {
            EmptyView()
        }
    }
}

private struct SettingsRow<Control: View>: View {
    let title: String
    let subtitle: String
    let controlWidth: CGFloat?
    private let control: Control

    init(
        title: String,
        subtitle: String,
        controlWidth: CGFloat? = nil,
        @ViewBuilder control: () -> Control
    ) {
        self.title = title
        self.subtitle = subtitle
        self.controlWidth = controlWidth
        self.control = control()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Group {
                if let controlWidth {
                    control
                        .frame(width: controlWidth, alignment: .trailing)
                } else {
                    control
                        .fixedSize()
                }
            }
        }
        .padding(.vertical, 8)
    }
}

private struct SettingsRowDivider: View {
    var body: some View {
        Divider()
            .overlay(AppTheme.separator)
    }
}

private struct ModelFileRow: View {
    let title: String
    let summary: String
    let systemImage: String
    let installState: ModelInstallState
    let warmState: WarmState
    let lastError: String?
    let progress: InstallableDownloadProgress?
    let actionTitle: String?
    let action: () -> Void

    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        formatter.zeroPadsFractionDigits = false
        return formatter
    }()

    private static let durationFormatter: DateComponentsFormatter = {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute]
        formatter.unitsStyle = .abbreviated
        formatter.maximumUnitCount = 2
        return formatter
    }()

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            iconTile

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.title3.weight(.semibold))

                Text(summary)
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    StatusBadge(
                        title: installState.rawValue.capitalized,
                        color: installStateColor(for: installState)
                    )

                    if let warmBadgeTitle {
                        StatusBadge(
                            title: warmBadgeTitle,
                            color: warmStateColor(for: warmState)
                        )
                    }
                }

                if let progress, installState == .downloading {
                    VStack(alignment: .leading, spacing: 6) {
                        ProgressView(value: progress.fractionCompleted)
                            .progressViewStyle(.linear)
                            .tint(AppTheme.accent)

                        HStack(spacing: 10) {
                            Text(downloadProgressSummary(progress))
                            Text("•")
                            Text(downloadProgressTiming(progress))
                        }
                        .font(.caption)
                        .foregroundStyle(AppTheme.secondaryText)
                    }
                }

                if let lastError {
                    Label(lastError, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(AppTheme.destructive)
                }
            }

            Spacer(minLength: 0)

            if let actionTitle {
                Button(actionTitle, action: action)
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                    .padding(.top, 2)
            }
        }
        .padding(.vertical, 10)
    }

    private var iconTile: some View {
        Image(systemName: systemImage)
            .font(.system(size: 18, weight: .semibold))
            .foregroundStyle(AppTheme.accent)
            .frame(width: 42, height: 42)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(AppTheme.actionCardFill)
            )
    }

    private var warmBadgeTitle: String? {
        switch warmState {
        case .cold:
            return nil
        case .warming:
            return "Warming"
        case .warm:
            return "Warm"
        case .error:
            return "Error"
        }
    }

    private func installStateColor(for state: ModelInstallState) -> Color {
        switch state {
        case .missing:
            return AppTheme.warning
        case .downloading:
            return AppTheme.accent
        case .installed:
            return AppTheme.success
        case .failed:
            return AppTheme.destructive
        }
    }

    private func warmStateColor(for state: WarmState) -> Color {
        switch state {
        case .cold:
            return AppTheme.secondaryText
        case .warming:
            return AppTheme.accent
        case .warm:
            return AppTheme.success
        case .error:
            return AppTheme.destructive
        }
    }

    private func downloadProgressSummary(_ progress: InstallableDownloadProgress) -> String {
        guard progress.bytesTotal > 0 else {
            return "Preparing download…"
        }

        return "\(formattedByteCount(progress.bytesDownloaded)) of \(formattedByteCount(progress.bytesTotal))"
    }

    private func downloadProgressTiming(_ progress: InstallableDownloadProgress) -> String {
        if let etaSeconds = progress.etaSeconds,
           etaSeconds.isFinite,
           etaSeconds > 1,
           let formattedETA = Self.durationFormatter.string(from: etaSeconds) {
            return "About \(formattedETA) left"
        }

        if let speedBytesPerSecond = progress.speedBytesPerSecond,
           speedBytesPerSecond.isFinite,
           speedBytesPerSecond > 0 {
            return "\(formattedByteCount(Int64(speedBytesPerSecond)))/s"
        }

        return progress.bytesTotal > 0 ? "Calculating time…" : "Estimating size…"
    }

    private func formattedByteCount(_ bytes: Int64) -> String {
        Self.byteFormatter.string(fromByteCount: bytes)
    }
}

private struct ModelFileEntry: Identifiable {
    let id: String
    let title: String
    let summary: String
    let systemImage: String
    let installState: ModelInstallState
    let warmState: WarmState
    let lastError: String?
    let progress: InstallableDownloadProgress?
    let actionTitle: String?
    let action: () -> Void
}

private struct StatusBadge: View {
    let title: String
    let color: Color

    var body: some View {
        Text(title)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(color.opacity(0.14), in: Capsule())
    }
}
