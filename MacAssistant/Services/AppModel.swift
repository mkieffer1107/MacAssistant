import AVFoundation
import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class AppModel {
    static let defaultImagePrompt = "Describe this image."

    enum Phase {
        case bootstrapping
        case setup
        case loading
        case ready
    }

    enum TranscriptSpeechPhase {
        case idle
        case synthesizing
        case playing
    }

    enum ReplySpeechPhase {
        case idle
        case buffering
        case playing
    }

    private struct VoiceRecordingState {
        enum Phase {
            case capturing
            case finalizing
        }

        let turnID: String
        let draftID: UUID
        let imageAttachment: ConversationImageAttachment?
        var phase: Phase
        var runtimeStarted = false
        var captureStartRetryCount = 0
        var hasObservedInputBuffer = false
        var isWaitingForAudio = false
        var capturedChunkCount = 0
        var bufferedChunks: [MicrophoneCaptureService.CaptureChunk] = []
    }

    private struct PendingAssistantDelta {
        var text: String = ""
        var isFinalSegment: Bool?
    }

    struct SettingsState: Codable, Sendable, Equatable {
        private enum CodingKeys: String, CodingKey {
            case defaultVoicePreset
            case inputDevicePreference
            case launchOnOpen
            case alwaysAcceptToolCalls
        }

        enum InputDevicePreference: Codable, Sendable, Equatable {
            case automatic
            case specificDeviceUID(String)

            private enum CodingKeys: String, CodingKey {
                case mode
                case uid
            }

            private enum Mode: String, Codable {
                case automatic
                case specific
            }

            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                let mode = try container.decodeIfPresent(Mode.self, forKey: .mode) ?? .automatic
                switch mode {
                case .automatic:
                    self = .automatic
                case .specific:
                    let uid = try container.decode(String.self, forKey: .uid)
                    self = .specificDeviceUID(uid)
                }
            }

            func encode(to encoder: Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                switch self {
                case .automatic:
                    try container.encode(Mode.automatic, forKey: .mode)
                case .specificDeviceUID(let uid):
                    try container.encode(Mode.specific, forKey: .mode)
                    try container.encode(uid, forKey: .uid)
                }
            }
        }

        var defaultVoicePreset: String
        var inputDevicePreference: InputDevicePreference
        var launchOnOpen: Bool
        var alwaysAcceptToolCalls: Bool

        init(
            defaultVoicePreset: String = "casual_male",
            inputDevicePreference: InputDevicePreference = .automatic,
            launchOnOpen: Bool = false,
            alwaysAcceptToolCalls: Bool = false
        ) {
            self.defaultVoicePreset = defaultVoicePreset
            self.inputDevicePreference = inputDevicePreference
            self.launchOnOpen = launchOnOpen
            self.alwaysAcceptToolCalls = alwaysAcceptToolCalls
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.defaultVoicePreset = try container.decodeIfPresent(String.self, forKey: .defaultVoicePreset) ?? "casual_male"
            self.inputDevicePreference = try container.decodeIfPresent(InputDevicePreference.self, forKey: .inputDevicePreference) ?? .automatic
            self.launchOnOpen = try container.decodeIfPresent(Bool.self, forKey: .launchOnOpen) ?? false
            self.alwaysAcceptToolCalls = try container.decodeIfPresent(Bool.self, forKey: .alwaysAcceptToolCalls) ?? false
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(defaultVoicePreset, forKey: .defaultVoicePreset)
            try container.encode(inputDevicePreference, forKey: .inputDevicePreference)
            try container.encode(launchOnOpen, forKey: .launchOnOpen)
            try container.encode(alwaysAcceptToolCalls, forKey: .alwaysAcceptToolCalls)
        }
    }

    private let appSupportURL: URL
    private let shouldPersistSettings: Bool
    @ObservationIgnored private let runtime: any RuntimeClienting
    @ObservationIgnored private let microphoneService: any MicrophoneCaptureServicing & Sendable
    @ObservationIgnored private let microphoneRouteCoordinator: any MicrophoneRouteCoordinating
    @ObservationIgnored private let playbackService: any AudioPlaybackServicing
    private let microphoneStartRetryDelay: Duration
    private let assistantDeltaFlushDelay: Duration
    private static let retryableMicrophoneStartErrorCode = -10868
    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        formatter.zeroPadsFractionDigits = false
        return formatter
    }()

    var phase: Phase = .bootstrapping
    var bootstrapStage = "Starting runtime…"
    var statusText = "Preparing runtime…"
    var bootstrapMessages: [String] = ["Opening the bundled runtime helper."]
    var modelInstallables = ModelInstallable.defaults
    var downloadProgressByInstallable: [String: InstallableDownloadProgress] = [:]
    var underlyingModels: [String: UnderlyingModelState] = Dictionary(
        uniqueKeysWithValues: RuntimeModelID.allCases.map {
            ($0.rawValue, UnderlyingModelState(id: $0.rawValue, installState: .missing, warmState: .cold, lastError: nil))
        }
    )
    var conversation: [ConversationItem] = []
    var transcriptRevision = 0
    var composerText = ""
    var pendingComposerImage: ComposerImageAttachment?
    var availableInputDevices: [MicrophoneCaptureService.InputDevice] = []
    var isMicReady = false
    var isStopping = false
    var runtimeLogs: [String] = []
    var permissionsSummary = "Microphone access is required. Automation and Accessibility permissions are granted by macOS when tool calls run."
    var showingSettings = false
    var settings = SettingsState() {
        didSet {
            persistSettingsIfNeeded()
        }
    }

    var activeDownloadProgress: InstallableDownloadProgress? {
        let activeProgress = modelInstallables
            .filter { $0.installState == .downloading }
            .compactMap { downloadProgressByInstallable[$0.id] }

        guard !activeProgress.isEmpty else { return nil }

        let bytesDownloaded = activeProgress.reduce(Int64(0)) { $0 + $1.bytesDownloaded }
        let bytesTotal = activeProgress.reduce(Int64(0)) { $0 + $1.bytesTotal }
        let totalSpeed = activeProgress.reduce(0.0) { partialResult, progress in
            partialResult + max(progress.speedBytesPerSecond ?? 0, 0)
        }
        let etaSeconds: Double?
        if bytesTotal > 0, totalSpeed > 0 {
            etaSeconds = Double(max(bytesTotal - bytesDownloaded, 0)) / totalSpeed
        } else {
            etaSeconds = nil
        }

        return InstallableDownloadProgress(
            installableID: "all_downloads",
            modelID: nil,
            bytesDownloaded: bytesDownloaded,
            bytesTotal: bytesTotal,
            speedBytesPerSecond: totalSpeed > 0 ? totalSpeed : nil,
            etaSeconds: etaSeconds,
            message: activeProgress.count == 1 ? activeProgress[0].message : "\(activeProgress.count) downloads in progress"
        )
    }

    func downloadProgress(for installableID: String) -> InstallableDownloadProgress? {
        downloadProgressByInstallable[installableID]
    }

    var lastRuntimeCommand: RuntimeCommandEnvelope? {
        runtime.lastSentCommand
    }

    var isAwaitingAgentResponse: Bool {
        if let activeResponseTurnID {
            return activeTurnIDs.contains(activeResponseTurnID)
        }
        return false
    }

    var isRecording: Bool {
        voiceRecordingState?.phase == .capturing
    }

    var hasLiveVoiceDraft: Bool {
        voiceRecordingState != nil
    }

    var isFinalizingVoiceRecording: Bool {
        voiceRecordingState?.phase == .finalizing
    }

    var isWaitingForMicrophoneAudio: Bool {
        voiceRecordingState?.isWaitingForAudio == true
    }

    var isReplySpeechActive: Bool {
        activeReplySpeechTurnID != nil
    }

    var canStopCurrentOutput: Bool {
        isAwaitingAgentResponse || isReplySpeechActive
    }

    var isComposerLocked: Bool {
        canStopCurrentOutput || hasLiveVoiceDraft
    }

    var isTranscriptActionsDisabled: Bool {
        hasLiveVoiceDraft || canStopCurrentOutput
    }

    var selectedInputDevicePickerValue: String {
        switch settings.inputDevicePreference {
        case .automatic:
            return ""
        case .specificDeviceUID(let uid):
            return uid
        }
    }

    var unavailableSelectedInputDeviceUID: String? {
        guard case .specificDeviceUID(let uid) = settings.inputDevicePreference else { return nil }
        return availableInputDevices.contains(where: { $0.uid == uid }) ? nil : uid
    }

    private var activeTurnIDs = Set<String>()
    private var activeResponseTurnID: String?
    private var activeSpeechTurnID: String?
    private var activeSpeechSourceAssistantSegmentID: String?
    private var transcriptSpeechPhase: TranscriptSpeechPhase = .idle
    private var activeReplySpeechTurnID: String?
    private var replySpeechPhase: ReplySpeechPhase = .idle
    private var speechOnlyTurnIDs = Set<String>()
    private var assistantMessageIDsBySegment: [String: Int] = [:]
    private var voiceDraftIDsByTurn: [String: Int] = [:]
    private var toolMessageIDsByCall: [String: Int] = [:]
    private var pendingAssistantDeltasBySegment: [String: PendingAssistantDelta] = [:]
    private var discardedTurnIDs = Set<String>()
    private var hasRequestedWarmup = false
    private var hasStartedBackgroundSTTWarm = false
    private var hasReceivedBootstrapProgress = false
    private var hasReceivedInitialModelState = false
    private var hasStarted = false
    private var isShuttingDown = false
    private let voiceCaptureWaitingDelay: Duration
    private var voiceRecordingState: VoiceRecordingState?
    @ObservationIgnored private var voiceCaptureWaitingTask: Task<Void, Never>?
    @ObservationIgnored private var microphoneStartRetryTask: Task<Void, Never>?
    @ObservationIgnored private var microphoneRoutePreflightTask: Task<Void, Never>?
    @ObservationIgnored private var microphoneAccessTask: Task<Void, Never>?
    @ObservationIgnored private var assistantDeltaFlushTask: Task<Void, Never>?

    private var attachmentsDirectoryURL: URL {
        appSupportURL.appendingPathComponent("attachments", isDirectory: true)
    }

    init(
        appSupportURL: URL? = nil,
        persistSettings: Bool? = nil,
        runtime: (any RuntimeClienting)? = nil,
        microphoneService: (any MicrophoneCaptureServicing & Sendable)? = nil,
        microphoneRouteCoordinator: (any MicrophoneRouteCoordinating)? = nil,
        playbackService: (any AudioPlaybackServicing)? = nil,
        voiceCaptureWaitingDelay: Duration = .seconds(2),
        microphoneStartRetryDelay: Duration = .milliseconds(250),
        assistantDeltaFlushDelay: Duration = .milliseconds(50)
    ) {
        let supportURL = appSupportURL ?? Self.defaultAppSupportURL()
        let shouldPersistSettings = persistSettings ?? !Self.isRunningTests
        self.appSupportURL = supportURL
        self.shouldPersistSettings = shouldPersistSettings
        self.runtime = runtime ?? AgentRuntimeClient(appSupportURL: supportURL)
        self.microphoneService = microphoneService ?? MicrophoneCaptureService()
        self.microphoneRouteCoordinator = microphoneRouteCoordinator ?? (Self.isRunningTests ? NoopMicrophoneRouteCoordinator.shared : MicrophoneRouteCoordinator.shared)
        self.playbackService = playbackService ?? AudioPlaybackService()
        self.voiceCaptureWaitingDelay = voiceCaptureWaitingDelay
        self.microphoneStartRetryDelay = microphoneStartRetryDelay
        self.assistantDeltaFlushDelay = assistantDeltaFlushDelay
        self.settings = shouldPersistSettings ? Self.loadSettings(from: supportURL) : SettingsState()
        self.availableInputDevices = self.microphoneService.availableInputDevices()
        self.runtime.onEvent = { [weak self] event in
            self?.handle(event: event)
        }
        self.runtime.onLog = { [weak self] message in
            self?.appendRuntimeLog(message)
        }
        self.runtime.onError = { [weak self] message in
            self?.appendRuntimeLog(message)
            self?.statusText = message
        }
        self.playbackService.onPlaybackStarted = { [weak self] in
            self?.handlePlaybackStarted()
        }
        self.playbackService.onPlaybackFinished = { [weak self] in
            self?.handlePlaybackFinished()
        }
        self.runtime.onTermination = { [weak self] message in
            guard let self else { return }
            if self.isShuttingDown { return }
            self.appendRuntimeLog(message)
            self.phase = .setup
            self.statusText = message
        }
        self.microphoneService.onInputDevicesChanged = { [weak self] in
            Task { @MainActor in
                self?.refreshAvailableInputDevices(logChange: true)
            }
        }
    }

    private static var isRunningTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
            || ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil
    }

    private static func defaultAppSupportURL() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MacAssistant", isDirectory: true)
    }

    private static func settingsFileURL(in appSupportURL: URL) -> URL {
        appSupportURL.appendingPathComponent("settings.json", isDirectory: false)
    }

    private static func loadSettings(from appSupportURL: URL) -> SettingsState {
        let fileURL = settingsFileURL(in: appSupportURL)
        guard
            let data = try? Data(contentsOf: fileURL),
            let decoded = try? JSONDecoder().decode(SettingsState.self, from: data)
        else {
            return SettingsState()
        }
        return decoded
    }

    private func persistSettingsIfNeeded() {
        guard shouldPersistSettings else { return }
        let fileURL = Self.settingsFileURL(in: appSupportURL)
        do {
            try FileManager.default.createDirectory(at: appSupportURL, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(settings)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            appendRuntimeLog("Failed to save settings: \(error.localizedDescription)")
        }
    }

    func start() {
        guard !hasStarted else { return }
        hasStarted = true
        hasReceivedBootstrapProgress = false
        hasReceivedInitialModelState = false
        bootstrapStage = "Starting runtime…"
        statusText = "Opening the bundled runtime helper."
        bootstrapMessages = ["Opening the bundled runtime helper."]
        do {
            try FileManager.default.createDirectory(at: appSupportURL, withIntermediateDirectories: true)
            try runtime.start()
            try runtime.send(RuntimeCommandEnvelope(
                type: "bootstrap",
                turnID: nil,
                modelID: nil,
                installableID: nil,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
            Task { [weak self] in
                try? await Task.sleep(for: .seconds(8))
                await MainActor.run {
                    guard let self, !self.hasReceivedBootstrapProgress, self.phase == .bootstrapping else { return }
                    self.bootstrapStage = "Still starting runtime…"
                    self.statusText = "The helper has not responded yet. This usually means macOS cannot find Bun or uv from the app environment."
                    self.appendBootstrapMessage("Still waiting for the runtime helper - checking whether Bun and uv are visible to the app.")
                }
            }
        } catch {
            statusText = error.localizedDescription
            phase = .setup
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func refreshAvailableInputDevices(logChange: Bool = false) {
        let previousDevices = availableInputDevices
        let updatedDevices = microphoneService.availableInputDevices()
        availableInputDevices = updatedDevices

        guard logChange, previousDevices != updatedDevices else { return }
        let summary = updatedDevices.isEmpty
            ? "none"
            : updatedDevices.map { device in
                "\(device.name) [\(device.uid)]"
            }.joined(separator: ", ")
        appendRuntimeLog("[Mic] Available input devices updated: \(summary)")
    }

    func setSelectedInputDevicePickerValue(_ value: String) {
        settings.inputDevicePreference = value.isEmpty ? .automatic : .specificDeviceUID(value)
    }

    func shutdown() {
        guard !isShuttingDown else { return }
        isShuttingDown = true

        microphoneAccessTask?.cancel()
        microphoneAccessTask = nil
        cancelMicrophoneRoutePreflight()
        cancelVoiceCaptureWaitingTask()
        cancelMicrophoneStartRetryTask()
        microphoneService.stop()
        microphoneRouteCoordinator.leaveRecordingRoute()
        voiceRecordingState = nil
        playbackService.stop()
        activeTurnIDs.removeAll()
        activeResponseTurnID = nil
        resetReplySpeechState()
        resetTranscriptSpeechState()
        pendingVoiceDraftCleanup()
        runtime.stop(gracefully: true)
    }

    func install(_ installable: ModelInstallable) {
        let previousStates = Dictionary(uniqueKeysWithValues: installable.modelIDs.map { modelID in
            let currentState = underlyingModels[modelID]
                ?? UnderlyingModelState(id: modelID, installState: .missing, warmState: .cold, lastError: nil)
            return (modelID, currentState)
        })

        for modelID in installable.modelIDs {
            underlyingModels[modelID] = UnderlyingModelState(
                id: modelID,
                installState: .downloading,
                warmState: .cold,
                lastError: nil
            )
        }
        downloadProgressByInstallable[installable.id] = InstallableDownloadProgress(
            installableID: installable.id,
            modelID: installable.modelIDs.first,
            bytesDownloaded: 0,
            bytesTotal: 0,
            speedBytesPerSecond: nil,
            etaSeconds: nil,
            message: "Preparing download…"
        )
        recomputeInstallables()
        statusText = "Downloading \(installable.title)…"

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "download_model",
                turnID: nil,
                modelID: nil,
                installableID: installable.id,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
        } catch {
            for (modelID, state) in previousStates {
                underlyingModels[modelID] = state
            }
            downloadProgressByInstallable.removeValue(forKey: installable.id)
            recomputeInstallables()
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func delete(_ installable: ModelInstallable) {
        downloadProgressByInstallable.removeValue(forKey: installable.id)
        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "delete_model",
                turnID: nil,
                modelID: nil,
                installableID: installable.id,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func deleteModel(_ modelID: RuntimeModelID) {
        let rawModelID = modelID.rawValue
        let previousState = underlyingModels[rawModelID]
            ?? UnderlyingModelState(id: rawModelID, installState: .missing, warmState: .cold, lastError: nil)

        for installable in modelInstallables where installable.modelIDs.contains(rawModelID) {
            downloadProgressByInstallable.removeValue(forKey: installable.id)
        }

        underlyingModels[rawModelID] = UnderlyingModelState(
            id: rawModelID,
            installState: .missing,
            warmState: .cold,
            lastError: nil
        )
        recomputeInstallables()
        statusText = "Deleting \(modelID.title)…"

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "delete_underlying_model",
                turnID: nil,
                modelID: rawModelID,
                installableID: nil,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
        } catch {
            underlyingModels[rawModelID] = previousState
            recomputeInstallables()
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func openModelsFolderInFinder() {
        let modelsURL = appSupportURL.appendingPathComponent("models", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: modelsURL, withIntermediateDirectories: true)
            NSWorkspace.shared.open(modelsURL)
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func promptForImageAttachment() {
        guard !isComposerLocked else { return }

        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true

        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                self?.importImage(from: url)
            }
        }
    }

    func importDroppedImage(from providers: [NSItemProvider]) -> Bool {
        guard !isComposerLocked else { return false }
        guard let provider = providers.first(where: Self.supportsImageDrop) else { return false }

        Task { @MainActor in
            do {
                try await self.importImage(from: provider)
            } catch {
                self.appendSystemStatus("Failed to import dropped image: \(error.localizedDescription)", level: "error")
            }
        }
        return true
    }

    func removePendingComposerImage() {
        guard !isComposerLocked else { return }
        pendingComposerImage = nil
    }

    func sendTextMessage() {
        guard !isComposerLocked else { return }
        let trimmedText = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        let imageAttachment = makeConversationImageAttachment(from: pendingComposerImage)
        let text = resolvedComposerText(from: trimmedText, imageAttachment: imageAttachment)
        guard !text.isEmpty || imageAttachment != nil else { return }
        cancelActiveSpeechPlaybackIfNeeded()

        let turnID = UUID().uuidString
        activeTurnIDs.insert(turnID)
        activeResponseTurnID = turnID
        conversation.append(.userMessage(UserMessage(
            id: UUID(),
            turnID: turnID,
            text: text,
            imageAttachment: imageAttachment,
            source: .typed,
            isCancelled: false
        )))
        noteConversationChanged()
        composerText = ""
        pendingComposerImage = nil
        statusText = "Thinking…"

        do {
            try sendTextTurn(
                turnID: turnID,
                text: text,
                imageAttachment: imageAttachment,
                speakReply: false
            )
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func sendCurrentInput() {
        if isRecording {
            stopRecording()
        } else if hasLiveVoiceDraft {
            return
        } else {
            sendTextMessage()
        }
    }

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else if !hasLiveVoiceDraft {
            startRecording()
        }
    }

    func startRecording() {
        guard
            voiceRecordingState == nil,
            microphoneAccessTask == nil,
            !canStopCurrentOutput,
            isMicReady || currentWarmState(for: .sttModel) == .warm
        else {
            return
        }

        cancelActiveSpeechPlaybackIfNeeded()

        switch microphoneService.authorizationStatus() {
        case .authorized:
            beginVoiceRecording()
        case .notDetermined:
            statusText = "Requesting microphone access…"
            microphoneAccessTask = Task { @MainActor [weak self] in
                guard let self else { return }

                let granted = await self.microphoneService.requestAccess()
                self.microphoneAccessTask = nil
                guard !Task.isCancelled else { return }
                guard self.voiceRecordingState == nil, !self.isAwaitingAgentResponse else { return }
                if granted {
                    self.beginVoiceRecording()
                } else {
                    let message = MicrophoneCaptureService.CaptureError.permissionDenied.localizedDescription
                    self.appendSystemStatus(message, level: "error")
                    self.statusText = message
                }
            }
        case .denied:
            let message = MicrophoneCaptureService.CaptureError.permissionDenied.localizedDescription
            appendSystemStatus(message, level: "error")
            statusText = message
        }
    }

    func stopRecording() {
        guard var state = voiceRecordingState, state.phase == .capturing else { return }

        cancelMicrophoneRoutePreflight()
        cancelMicrophoneStartRetryTask()
        microphoneService.stop()
        microphoneRouteCoordinator.leaveRecordingRoute()
        cancelVoiceCaptureWaitingTask()
        state.isWaitingForAudio = false

        guard state.runtimeStarted, state.capturedChunkCount > 0 else {
            if state.imageAttachment != nil {
                submitAttachmentOnlyVoiceRecording(state)
                return
            }
            failVoiceRecording(
                turnID: state.turnID,
                message: "No microphone audio was captured. Try speaking again.",
                restorePendingImage: false,
                cancelRuntime: false
            )
            return
        }

        state.phase = .finalizing
        voiceRecordingState = state
        activeResponseTurnID = state.turnID
        statusText = "Finalizing speech…"

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "stop_recording",
                turnID: state.turnID,
                modelID: nil,
                installableID: nil,
                text: nil,
                speakReply: true,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
        } catch {
            failVoiceRecording(
                turnID: state.turnID,
                message: error.localizedDescription,
                restorePendingImage: false,
                cancelRuntime: state.runtimeStarted
            )
        }
    }

    func stopActiveTurn() {
        if hasLiveVoiceDraft {
            discardCurrentInput()
            return
        }

        if let turnID = activeResponseTurnID {
            isStopping = true
            playbackService.stop()
            resetReplySpeechState()
            do {
                try runtime.send(RuntimeCommandEnvelope(
                    type: "cancel",
                    turnID: turnID,
                    modelID: nil,
                    installableID: nil,
                    text: nil,
                    speakReply: nil,
                    chunkBase64: nil,
                    sampleRate: nil,
                    channels: nil,
                    toolCallID: nil,
                    attachments: nil,
                    arguments: nil
                ))
            } catch {
                appendSystemStatus(error.localizedDescription, level: "error")
            }
            return
        }

        if isReplySpeechActive {
            playbackService.stop()
            resetReplySpeechState()
            clearReadyStatusAfterSpeechIfNeeded()
            return
        }

        cancelActiveSpeechPlaybackIfNeeded()
    }

    func stopEverything() {
        stopActiveTurn()
        cancelActiveSpeechPlaybackIfNeeded()
    }

    var canResetConversation: Bool {
        hasLiveVoiceDraft
            || pendingComposerImage != nil
            || !composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !conversation.isEmpty
    }

    func resetConversation() {
        let turnIDsToIgnore = activeTurnIDs
        discardedTurnIDs.formUnion(turnIDsToIgnore)
        if let state = voiceRecordingState, state.runtimeStarted {
            discardedTurnIDs.insert(state.turnID)
        }

        microphoneAccessTask?.cancel()
        microphoneAccessTask = nil
        cancelMicrophoneRoutePreflight()
        cancelVoiceCaptureWaitingTask()
        cancelMicrophoneStartRetryTask()
        microphoneService.stop()
        microphoneRouteCoordinator.leaveRecordingRoute()
        voiceRecordingState = nil
        cancelActiveSpeechPlaybackIfNeeded()
        playbackService.stop()
        isStopping = false
        composerText = ""
        pendingComposerImage = nil
        clearPendingAssistantDeltaState()
        conversation.removeAll()
        noteConversationChanged()
        activeTurnIDs.removeAll()
        activeResponseTurnID = nil
        resetReplySpeechState()
        resetTranscriptSpeechState()
        speechOnlyTurnIDs.removeAll()
        assistantMessageIDsBySegment.removeAll()
        voiceDraftIDsByTurn.removeAll()
        toolMessageIDsByCall.removeAll()
        statusText = "Ready"

        guard hasStarted else { return }

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "reset_conversation",
                turnID: nil,
                modelID: nil,
                installableID: nil,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func discardCurrentInput() {
        guard let state = voiceRecordingState else { return }
        discardVoiceRecording(
            state,
            restorePendingImage: true,
            cancelRuntime: state.runtimeStarted,
            markDiscarded: state.runtimeStarted
        )
        statusText = "Ready"
    }

    private func submitAttachmentOnlyVoiceRecording(_ state: VoiceRecordingState) {
        cancelVoiceCaptureWaitingTask()
        voiceRecordingState = nil

        guard commitVoiceDraft(turnID: state.turnID, text: "") else {
            removeVoiceDraft(turnID: state.turnID)
            statusText = "Ready"
            return
        }

        let submittedText = resolvedComposerText(from: "", imageAttachment: state.imageAttachment)
        activeTurnIDs.insert(state.turnID)
        activeResponseTurnID = state.turnID
        statusText = "Thinking…"

        do {
            try sendTextTurn(
                turnID: state.turnID,
                text: submittedText,
                imageAttachment: state.imageAttachment,
                speakReply: true
            )
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
            statusText = error.localizedDescription
        }
    }

    private func beginVoiceRecording() {
        refreshAvailableInputDevices()
        let turnID = UUID().uuidString
        let draftID = UUID()
        let imageAttachment = makeConversationImageAttachment(from: pendingComposerImage)
        voiceRecordingState = VoiceRecordingState(
            turnID: turnID,
            draftID: draftID,
            imageAttachment: imageAttachment,
            phase: .capturing
        )
        conversation.append(.userVoiceDraft(UserVoiceDraft(
            id: draftID,
            turnID: turnID,
            text: "",
            imageAttachment: imageAttachment,
            source: .voice,
            isCancelled: false
        )))
        voiceDraftIDsByTurn[turnID] = conversation.count - 1
        noteConversationChanged()
        pendingComposerImage = nil
        statusText = "Listening…"
        startMicrophoneCapture(for: turnID, isRetry: false)
    }

    private func handleMicrophoneCaptureStarted(turnID: String) {
        guard voiceRecordingState?.turnID == turnID else { return }
        statusText = "Listening…"
    }

    private func handleMicrophoneInputBuffer(turnID: String) {
        registerMicrophoneActivity(turnID: turnID)
    }

    private func handleMicrophoneFirstChunk(turnID: String) {
        registerMicrophoneActivity(turnID: turnID)
    }

    private func registerMicrophoneActivity(turnID: String) {
        guard var state = voiceRecordingState, state.turnID == turnID, state.phase == .capturing else { return }
        let wasWaiting = state.isWaitingForAudio
        state.hasObservedInputBuffer = true
        state.isWaitingForAudio = false
        voiceRecordingState = state
        cancelVoiceCaptureWaitingTask()
        if wasWaiting {
            statusText = "Listening…"
        }
    }

    private func handleCapturedAudioChunk(_ chunk: MicrophoneCaptureService.CaptureChunk, turnID: String) {
        guard var state = voiceRecordingState, state.turnID == turnID, state.phase == .capturing else { return }

        state.hasObservedInputBuffer = true
        state.isWaitingForAudio = false
        state.capturedChunkCount += 1
        if !state.runtimeStarted {
            state.bufferedChunks.append(chunk)
            voiceRecordingState = state
            cancelVoiceCaptureWaitingTask()
            do {
                try startRuntimeVoiceTurnIfNeeded(for: turnID)
            } catch {
                let cancelRuntime = voiceRecordingState?.turnID == turnID && voiceRecordingState?.runtimeStarted == true
                failVoiceRecording(
                    turnID: turnID,
                    message: error.localizedDescription,
                    restorePendingImage: true,
                    cancelRuntime: cancelRuntime
                )
            }
            return
        }

        voiceRecordingState = state
        cancelVoiceCaptureWaitingTask()
        do {
            try sendAudioChunk(chunk, turnID: turnID)
        } catch {
            failVoiceRecording(
                turnID: turnID,
                message: error.localizedDescription,
                restorePendingImage: true,
                cancelRuntime: true
            )
        }
    }

    private func startRuntimeVoiceTurnIfNeeded(for turnID: String) throws {
        guard var state = voiceRecordingState, state.turnID == turnID, !state.runtimeStarted else { return }

        try runtime.send(RuntimeCommandEnvelope(
            type: "start_recording",
            turnID: turnID,
            modelID: nil,
            installableID: nil,
            text: nil,
            speakReply: true,
            chunkBase64: nil,
            sampleRate: nil,
            channels: nil,
            toolCallID: nil,
            attachments: runtimeAttachments(from: state.imageAttachment),
            arguments: ["voice_preset": .string(settings.defaultVoicePreset)]
        ))

        activeTurnIDs.insert(turnID)
        state.runtimeStarted = true
        let bufferedChunks = state.bufferedChunks
        state.bufferedChunks.removeAll(keepingCapacity: false)
        voiceRecordingState = state

        for chunk in bufferedChunks {
            try sendAudioChunk(chunk, turnID: turnID)
        }
    }

    private func handleMicrophoneCaptureError(turnID: String, error: Error) {
        guard let state = voiceRecordingState, state.turnID == turnID else { return }
        failVoiceRecording(
            turnID: turnID,
            message: error.localizedDescription,
            restorePendingImage: true,
            cancelRuntime: state.runtimeStarted
        )
    }

    private func handleWaitingForMicrophoneAudio(turnID: String) {
        guard
            var state = voiceRecordingState,
            state.turnID == turnID,
            state.phase == .capturing,
            state.capturedChunkCount == 0,
            !state.hasObservedInputBuffer
        else {
            return
        }

        state.isWaitingForAudio = true
        voiceRecordingState = state
        statusText = "Waiting for microphone audio…"
        appendRuntimeLog("[Mic] Waiting for microphone audio after \(Self.describe(duration: voiceCaptureWaitingDelay)) without input.")
    }

    private func resolvedPreferredInputDevice() -> MicrophoneCaptureService.InputDevice? {
        let currentDevices = availableInputDevices.isEmpty ? microphoneService.availableInputDevices() : availableInputDevices

        switch settings.inputDevicePreference {
        case .automatic:
            if let builtInDevice = currentDevices.first(where: { $0.isBuiltIn }) {
                return builtInDevice
            }
            if let defaultInputDevice = microphoneService.defaultInputDevice(),
               let matchingDevice = currentDevices.first(where: { $0.uid == defaultInputDevice.uid }) {
                return matchingDevice
            }
            return currentDevices.first
        case .specificDeviceUID(let uid):
            if let selectedDevice = currentDevices.first(where: { $0.uid == uid }) {
                return selectedDevice
            }
            return currentDevices.first(where: { $0.isBuiltIn })
                ?? microphoneService.defaultInputDevice()
                ?? currentDevices.first
        }
    }

    private func describe(inputPreference: SettingsState.InputDevicePreference) -> String {
        switch inputPreference {
        case .automatic:
            return "automatic (built-in preferred)"
        case .specificDeviceUID(let uid):
            if let device = availableInputDevices.first(where: { $0.uid == uid }) {
                return "specific \(describe(inputDevice: device))"
            }
            return "specific unavailable [\(uid)]"
        }
    }

    private func describe(inputDevice: MicrophoneCaptureService.InputDevice?) -> String {
        guard let inputDevice else { return "system-default" }
        return "\(inputDevice.name) [\(inputDevice.uid)]"
    }

    private func describe(outputDevice: MicrophoneRouteCoordinator.OutputDevice?) -> String {
        guard let outputDevice else { return "system-default" }
        return "\(outputDevice.name) [\(outputDevice.uid)]"
    }

    private func scheduleVoiceCaptureWaitingTask(for turnID: String) {
        cancelVoiceCaptureWaitingTask()
        let delay = voiceCaptureWaitingDelay
        voiceCaptureWaitingTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                self?.handleWaitingForMicrophoneAudio(turnID: turnID)
            }
        }
    }

    private func cancelVoiceCaptureWaitingTask() {
        voiceCaptureWaitingTask?.cancel()
        voiceCaptureWaitingTask = nil
    }

    private func startMicrophoneCapture(for turnID: String, isRetry: Bool) {
        guard let state = voiceRecordingState, state.turnID == turnID, state.phase == .capturing else { return }
        cancelMicrophoneRoutePreflight()

        let resolvedInputDevice = resolvedPreferredInputDevice()
        let routeState = microphoneRouteCoordinator.routeStateSnapshot()
        if isRetry {
            appendRuntimeLog(
                "[Mic] Retrying capture preference=\(describe(inputPreference: settings.inputDevicePreference)) " +
                "resolved=\(describe(inputDevice: resolvedInputDevice)) " +
                "defaultOutput=\(describe(outputDevice: routeState.defaultOutputDevice))"
            )
        } else {
            appendRuntimeLog(
                "[Mic] Starting capture preference=\(describe(inputPreference: settings.inputDevicePreference)) " +
                "resolved=\(describe(inputDevice: resolvedInputDevice)) " +
                "defaultOutput=\(describe(outputDevice: routeState.defaultOutputDevice))"
            )
        }

        microphoneRoutePreflightTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.microphoneRoutePreflightTask = nil }

            do {
                let preflightResult = try await self.microphoneRouteCoordinator.prepareForRecording(inputDevice: resolvedInputDevice)
                guard !Task.isCancelled else {
                    self.microphoneRouteCoordinator.leaveRecordingRoute()
                    return
                }
                guard let activeState = self.voiceRecordingState, activeState.turnID == turnID, activeState.phase == .capturing else {
                    self.microphoneRouteCoordinator.leaveRecordingRoute()
                    return
                }

                self.appendRuntimeLog(
                    "[Mic] Split-route preflight \(preflightResult.didEngageArbitration ? "engaged" : "skipped") " +
                    "requestedInput=\(self.describe(inputDevice: resolvedInputDevice)) " +
                    "defaultOutput=\(self.describe(outputDevice: preflightResult.defaultOutputDevice))"
                )
                self.beginMicrophoneCaptureSession(
                    for: turnID,
                    resolvedInputDevice: resolvedInputDevice,
                    currentState: activeState
                )
            } catch is CancellationError {
                self.microphoneRouteCoordinator.leaveRecordingRoute()
            } catch {
                self.microphoneRouteCoordinator.leaveRecordingRoute()
                if self.handleRetryableCaptureStartFailure(
                    for: turnID,
                    error: error,
                    attemptedInputDevice: resolvedInputDevice,
                    currentState: state
                ) {
                    return
                }
                self.failVoiceRecording(
                    turnID: turnID,
                    message: error.localizedDescription,
                    restorePendingImage: true,
                    cancelRuntime: false
                )
            }
        }
    }

    private func beginMicrophoneCaptureSession(
        for turnID: String,
        resolvedInputDevice: MicrophoneCaptureService.InputDevice?,
        currentState: VoiceRecordingState
    ) {
        do {
            try microphoneService.start(
                preferredInputDeviceUID: resolvedInputDevice?.uid,
                onStarted: { [weak self] in
                    Task { @MainActor in
                        self?.handleMicrophoneCaptureStarted(turnID: turnID)
                    }
                },
                onFirstInputBuffer: { [weak self] in
                    Task { @MainActor in
                        self?.handleMicrophoneInputBuffer(turnID: turnID)
                    }
                },
                onFirstChunk: { [weak self] in
                    Task { @MainActor in
                        self?.handleMicrophoneFirstChunk(turnID: turnID)
                    }
                },
                onError: { [weak self] error in
                    Task { @MainActor in
                        self?.handleMicrophoneCaptureError(turnID: turnID, error: error)
                    }
                },
                logHandler: { [weak self] message in
                    Task { @MainActor in
                        self?.appendRuntimeLog(message)
                    }
                },
                chunkHandler: { [weak self] chunk in
                    Task { @MainActor in
                        self?.handleCapturedAudioChunk(chunk, turnID: turnID)
                    }
                }
            )
            scheduleVoiceCaptureWaitingTask(for: turnID)
        } catch {
            microphoneRouteCoordinator.leaveRecordingRoute()
            if handleRetryableCaptureStartFailure(
                for: turnID,
                error: error,
                attemptedInputDevice: resolvedInputDevice,
                currentState: currentState
            ) {
                return
            }
            failVoiceRecording(
                turnID: turnID,
                message: error.localizedDescription,
                restorePendingImage: true,
                cancelRuntime: false
            )
        }
    }

    private func handleRetryableCaptureStartFailure(
        for turnID: String,
        error: Error,
        attemptedInputDevice: MicrophoneCaptureService.InputDevice?,
        currentState: VoiceRecordingState
    ) -> Bool {
        guard isRetryableMicrophoneStartError(error) else { return false }
        guard currentState.captureStartRetryCount == 0 else { return false }
        guard !currentState.hasObservedInputBuffer, currentState.capturedChunkCount == 0 else { return false }
        guard var state = voiceRecordingState, state.turnID == turnID, state.phase == .capturing else { return false }

        state.captureStartRetryCount += 1
        voiceRecordingState = state
        refreshAvailableInputDevices()
        let retriedInputDevice = resolvedPreferredInputDevice()
        let routeState = microphoneRouteCoordinator.routeStateSnapshot()
        appendRuntimeLog(
            "[Mic] Capture start failed before audio arrived: \(error.localizedDescription). " +
            "Retrying in \(Self.describe(duration: microphoneStartRetryDelay)) " +
            "attempted=\(describe(inputDevice: attemptedInputDevice)) " +
            "resolved=\(describe(inputDevice: retriedInputDevice)) " +
            "defaultOutput=\(describe(outputDevice: routeState.defaultOutputDevice))"
        )
        scheduleMicrophoneStartRetry(for: turnID)
        return true
    }

    private func isRetryableMicrophoneStartError(_ error: Error) -> Bool {
        if error is MicrophoneRouteCoordinator.PreflightError {
            return true
        }
        let nsError = error as NSError
        return nsError.code == Self.retryableMicrophoneStartErrorCode
    }

    private func scheduleMicrophoneStartRetry(for turnID: String) {
        cancelMicrophoneStartRetryTask()
        let delay = microphoneStartRetryDelay
        microphoneStartRetryTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self else { return }
                self.microphoneStartRetryTask = nil
                guard let state = self.voiceRecordingState, state.turnID == turnID, state.phase == .capturing else { return }
                guard !state.hasObservedInputBuffer, state.capturedChunkCount == 0 else { return }
                self.startMicrophoneCapture(for: turnID, isRetry: true)
            }
        }
    }

    private func cancelMicrophoneStartRetryTask() {
        microphoneStartRetryTask?.cancel()
        microphoneStartRetryTask = nil
    }

    private func cancelMicrophoneRoutePreflight() {
        microphoneRoutePreflightTask?.cancel()
        microphoneRoutePreflightTask = nil
        microphoneRouteCoordinator.cancelPendingPreparation()
    }

    private func failVoiceRecording(
        turnID: String,
        message: String,
        restorePendingImage: Bool,
        cancelRuntime: Bool
    ) {
        microphoneRouteCoordinator.leaveRecordingRoute()
        if let state = voiceRecordingState, state.turnID == turnID {
            discardVoiceRecording(
                state,
                restorePendingImage: restorePendingImage,
                cancelRuntime: cancelRuntime,
                markDiscarded: cancelRuntime && state.runtimeStarted
            )
        }
        appendSystemStatus(message, level: "error")
        statusText = message
    }

    private func discardVoiceRecording(
        _ state: VoiceRecordingState,
        restorePendingImage: Bool,
        cancelRuntime: Bool,
        markDiscarded: Bool
    ) {
        microphoneAccessTask?.cancel()
        microphoneAccessTask = nil
        cancelMicrophoneRoutePreflight()
        cancelVoiceCaptureWaitingTask()
        cancelMicrophoneStartRetryTask()
        microphoneService.stop()
        microphoneRouteCoordinator.leaveRecordingRoute()
        if restorePendingImage {
            restorePendingComposerImage(from: state.imageAttachment)
        }
        if markDiscarded {
            discardedTurnIDs.insert(state.turnID)
        }
        voiceRecordingState = nil
        removeVoiceDraft(turnID: state.turnID)
        activeTurnIDs.remove(state.turnID)
        if activeResponseTurnID == state.turnID {
            activeResponseTurnID = nil
        }
        isStopping = false
        playbackService.stop()
        resetReplySpeechState()

        guard cancelRuntime else { return }

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "cancel",
                turnID: state.turnID,
                modelID: nil,
                installableID: nil,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func approveTool(_ tool: ToolInvocation) {
        approveTool(turnID: tool.turnID, toolCallID: tool.toolCallID)
    }

    private func approveTool(turnID: String, toolCallID: String, suppressRuntimeUnavailableError: Bool = false) {
        if suppressRuntimeUnavailableError, !runtime.isRunning {
            updateTool(toolCallID) { tool in
                tool.approvalState = .approved
                tool.executionState = .running
            }
            return
        }

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "approve_tool",
                turnID: turnID,
                modelID: nil,
                installableID: nil,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: toolCallID,
                attachments: nil,
                arguments: nil
            ))
            updateTool(toolCallID) { tool in
                tool.approvalState = .approved
                tool.executionState = .running
            }
        } catch {
            if suppressRuntimeUnavailableError,
               let runtimeError = error as? RuntimeLaunchError,
               runtimeError == .runtimeUnavailable {
                return
            }
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func denyTool(_ tool: ToolInvocation) {
        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "deny_tool",
                turnID: tool.turnID,
                modelID: nil,
                installableID: nil,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: tool.toolCallID,
                attachments: nil,
                arguments: nil
            ))
            updateTool(tool.toolCallID) { tool in
                tool.approvalState = .denied
                tool.executionState = .finished
                tool.output = "Tool execution denied by the user."
            }
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func toggleExpansion(for tool: ToolInvocation) {
        updateTool(tool.toolCallID) { tool in
            tool.isExpanded.toggle()
        }
    }

    func copyUserMessage(_ message: UserMessage) {
        copyTextToPasteboard(message.text)
    }

    func copyAssistantMessage(_ message: AssistantMessage) {
        copyTextToPasteboard(message.text)
    }

    func isSpeakingAssistantMessage(_ message: AssistantMessage) -> Bool {
        activeSpeechSourceAssistantSegmentID == message.segmentID && transcriptSpeechPhase != .idle
    }

    func canToggleSpeech(for message: AssistantMessage) -> Bool {
        if isSpeakingAssistantMessage(message) {
            return true
        }
        return !isTranscriptActionsDisabled && !message.isStreaming && !message.text.isEmpty
    }

    func speakAssistantMessage(_ message: AssistantMessage) {
        guard !message.text.isEmpty, !message.isStreaming else { return }
        if isSpeakingAssistantMessage(message) {
            cancelActiveSpeechPlaybackIfNeeded()
            return
        }
        guard !isTranscriptActionsDisabled else { return }
        cancelActiveSpeechPlaybackIfNeeded()

        let turnID = UUID().uuidString
        activeTurnIDs.insert(turnID)
        activeSpeechTurnID = turnID
        activeSpeechSourceAssistantSegmentID = message.segmentID
        transcriptSpeechPhase = .synthesizing
        speechOnlyTurnIDs.insert(turnID)
        statusText = "Speaking…"

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "speak_text",
                turnID: turnID,
                text: message.text,
                arguments: ["voice_preset": .string(settings.defaultVoicePreset)]
            ))
        } catch {
            activeTurnIDs.remove(turnID)
            speechOnlyTurnIDs.remove(turnID)
            resetTranscriptSpeechState()
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    func redoAssistantMessage(_ message: AssistantMessage) {
        guard !isTranscriptActionsDisabled else { return }
        rerunTurn(turnID: message.turnID, updatedText: nil, updatedSource: nil)
    }

    func editUserMessage(_ message: UserMessage, newText: String) {
        guard !isTranscriptActionsDisabled else { return }
        rerunTurn(turnID: message.turnID, updatedText: newText, updatedSource: .typed)
    }

    func copyRuntimeLogs() {
        copyTextToPasteboard(runtimeLogsText)
    }

    private func sendAudioChunk(_ chunk: MicrophoneCaptureService.CaptureChunk, turnID: String) throws {
        let base64 = chunk.data.base64EncodedString()
        try runtime.send(RuntimeCommandEnvelope(
            type: "append_audio_chunk",
            turnID: turnID,
            modelID: nil,
            installableID: nil,
            text: nil,
            speakReply: true,
            chunkBase64: base64,
            sampleRate: chunk.sampleRate,
            channels: chunk.channels,
            toolCallID: nil,
            attachments: nil,
            arguments: nil
        ))
    }

    var runtimeLogsText: String {
        runtimeLogs.joined(separator: "\n")
    }

    func handle(event: RuntimeEventEnvelope) {
        if event.type != "assistant_delta" {
            flushPendingAssistantDeltas()
        }

        switch event.type {
        case "bootstrap_progress":
            hasReceivedBootstrapProgress = true
            if !hasReceivedInitialModelState {
                phase = .bootstrapping
            }
            bootstrapStage = event.stage ?? "Preparing runtime…"
            statusText = event.message ?? bootstrapStage
            appendBootstrapMessage("\(bootstrapStage) - \(statusText)")
        case "model_state":
            handleModelState(event)
        case "download_progress":
            handleDownloadProgress(event)
        case "transcript_delta":
            guard let turnID = event.turnID, !discardedTurnIDs.contains(turnID) else { return }
            updateVoiceDraft(turnID: turnID, text: event.text ?? "")
        case "transcript_final":
            guard let turnID = event.turnID, !discardedTurnIDs.contains(turnID) else { return }
            if voiceRecordingState?.turnID == turnID {
                cancelVoiceCaptureWaitingTask()
                voiceRecordingState = nil
            }
            if commitVoiceDraft(turnID: turnID, text: event.text ?? "") {
                statusText = "Thinking…"
            }
        case "assistant_delta":
            guard let turnID = event.turnID, !discardedTurnIDs.contains(turnID) else { return }
            enqueueAssistantDelta(
                turnID: turnID,
                segmentID: event.assistantSegmentID ?? turnID,
                delta: event.text ?? "",
                isFinalSegment: event.isFinalSegment
            )
        case "tool_proposed":
            handleToolProposal(event)
        case "tool_started":
            guard let toolCallID = event.toolCallID else { return }
            updateTool(toolCallID) { tool in
                tool.executionState = .running
                tool.summary = event.toolSummary ?? tool.summary
            }
        case "tool_output":
            guard let toolCallID = event.toolCallID else { return }
            updateTool(toolCallID) { tool in
                tool.output = event.output ?? ""
                if !settings.alwaysAcceptToolCalls {
                    tool.isExpanded = true
                }
            }
        case "tool_finished":
            guard let toolCallID = event.toolCallID else { return }
            updateTool(toolCallID) { tool in
                tool.executionState = ExecutionState(rawValue: event.executionState ?? "") ?? .finished
                tool.output = event.output ?? tool.output
            }
        case "audio_chunk":
            if let turnID = event.turnID,
               turnID == activeResponseTurnID || turnID == activeSpeechTurnID || turnID == activeReplySpeechTurnID,
               let chunkBase64 = event.chunkBase64,
               let data = Data(base64Encoded: chunkBase64) {
                if turnID == activeResponseTurnID {
                    beginReplySpeechIfNeeded(for: turnID)
                }
                playbackService.enqueuePCMData(
                    data,
                    sampleRate: event.sampleRate ?? 24_000,
                    channels: event.channels ?? 1,
                    encoding: event.audioEncoding ?? "pcm_s16le"
                )
            }
        case "turn_finished":
            handleTurnFinished(event)
        case "error":
            appendSystemStatus(event.message ?? "Unknown runtime error", level: "error")
            statusText = event.message ?? "Runtime error"
        default:
            appendRuntimeLog("Unhandled event: \(event.type)")
        }
    }

    private func handleModelState(_ event: RuntimeEventEnvelope) {
        guard let modelID = event.modelID else { return }
        hasReceivedInitialModelState = true
        let installState = ModelInstallState(rawValue: event.installState ?? "") ?? .missing
        let warmState = WarmState(rawValue: event.warmState ?? "") ?? .cold
        underlyingModels[modelID] = UnderlyingModelState(
            id: modelID,
            installState: installState,
            warmState: warmState,
            lastError: event.message
        )
        recomputeInstallables()

        let coreWarmFailure = currentWarmState(for: .agentModel) == .error || currentWarmState(for: .ttsModel) == .error

        if coreWarmFailure {
            phase = .setup
            statusText = event.message ?? "A required model failed to load."
            hasRequestedWarmup = false
        } else if installablesReadyForChat, !hasRequestedWarmup {
            hasRequestedWarmup = true
            phase = .loading
            statusText = "Warming agent and voice models…"
            warmCoreModels()
        } else if !installablesReadyForChat {
            phase = .setup
        } else if currentWarmState(for: .agentModel) == .warm && currentWarmState(for: .ttsModel) == .warm {
            phase = .ready
            statusText = "Ready"
            if !hasStartedBackgroundSTTWarm {
                hasStartedBackgroundSTTWarm = true
                warmSTTInBackground()
            }
        } else if installablesReadyForChat {
            phase = .loading
        }

        isMicReady = currentWarmState(for: .sttModel) == .warm
    }

    private func handleDownloadProgress(_ event: RuntimeEventEnvelope) {
        guard let installableID = event.installableID else { return }

        let bytesDownloaded = max(event.bytesDownloaded ?? 0, 0)
        let bytesTotal = max(event.bytesTotal ?? 0, 0)
        let clampedDownloaded = bytesTotal > 0 ? min(bytesDownloaded, bytesTotal) : bytesDownloaded
        let progress = InstallableDownloadProgress(
            installableID: installableID,
            modelID: event.modelID,
            bytesDownloaded: clampedDownloaded,
            bytesTotal: bytesTotal,
            speedBytesPerSecond: event.speedBytesPerSecond,
            etaSeconds: event.etaSeconds,
            message: event.message
        )
        downloadProgressByInstallable[installableID] = progress

        if let installable = modelInstallables.first(where: { $0.id == installableID }) {
            statusText = downloadProgressStatusText(for: progress, title: installable.title)
        }
    }

    private func handleToolProposal(_ event: RuntimeEventEnvelope) {
        guard
            let turnID = event.turnID,
            let toolCallID = event.toolCallID,
            let toolName = event.toolName,
            !discardedTurnIDs.contains(turnID)
        else { return }
        let approvalState = ApprovalState(rawValue: event.approvalState ?? "") ?? .notRequired
        let tool = ToolInvocation(
            id: UUID(),
            turnID: turnID,
            toolCallID: toolCallID,
            name: toolName,
            arguments: event.toolArguments ?? [:],
            safetyClass: SafetyClass(rawValue: event.safetyClass ?? "") ?? .readOnly,
            approvalState: approvalState,
            executionState: ExecutionState(rawValue: event.executionState ?? "") ?? .proposed,
            summary: event.toolSummary ?? toolName,
            output: "",
            isExpanded: approvalState == .pending && !settings.alwaysAcceptToolCalls
        )
        conversation.append(.toolInvocation(tool))
        toolMessageIDsByCall[toolCallID] = conversation.count - 1
        noteConversationChanged()
        if settings.alwaysAcceptToolCalls, tool.approvalState == .pending {
            approveTool(turnID: turnID, toolCallID: toolCallID, suppressRuntimeUnavailableError: true)
        }
    }

    private func handleTurnFinished(_ event: RuntimeEventEnvelope) {
        guard let turnID = event.turnID else { return }
        let status = event.status ?? "finished"
        let wasDiscarded = discardedTurnIDs.contains(turnID)
        let wasActiveResponseTurn = activeResponseTurnID == turnID
        let wasActiveSpeechTurn = activeSpeechTurnID == turnID
        let wasActiveReplySpeechTurn = activeReplySpeechTurnID == turnID
        let wasSpeechOnlyTurn = speechOnlyTurnIDs.contains(turnID)
        let wasActiveVoiceTurn = voiceRecordingState?.turnID == turnID
        activeTurnIDs.remove(turnID)
        finishAssistantMessages(turnID)
        if wasActiveVoiceTurn {
            cancelVoiceCaptureWaitingTask()
            voiceRecordingState = nil
        }
        if wasActiveResponseTurn {
            activeResponseTurnID = nil
        }
        if wasActiveSpeechTurn {
            activeSpeechTurnID = nil
        }
        if wasActiveSpeechTurn {
            if status == "finished" {
                playbackService.finishStream()
            } else {
                playbackService.stop()
            }
        }
        if wasActiveResponseTurn {
            if status == "finished" {
                if wasActiveReplySpeechTurn {
                    playbackService.finishStream()
                    statusText = "Speaking…"
                }
            } else {
                playbackService.stop()
                resetReplySpeechState(turnID: turnID)
            }
        }
        if wasSpeechOnlyTurn {
            speechOnlyTurnIDs.remove(turnID)
            discardedTurnIDs.remove(turnID)
            if status == "failed" {
                resetTranscriptSpeechState()
                appendSystemStatus(event.message ?? "Speech failed", level: "error")
            } else if status == "cancelled" {
                resetTranscriptSpeechState()
            }
            return
        }
        if wasDiscarded {
            if event.isFinal == true || status == "cancelled" || status == "failed" || status == "finished" {
                removeVoiceDraft(turnID: turnID)
                discardedTurnIDs.remove(turnID)
                isStopping = false
                statusText = "Ready"
            }
            return
        }
        if voiceDraftIDsByTurn[turnID] != nil {
            if status == "finished" {
                _ = commitVoiceDraft(turnID: turnID, text: "")
            } else {
                removeVoiceDraft(turnID: turnID)
            }
        }
        if status == "cancelled" {
            markTurnCancelled(turnID)
            statusText = "Stopped"
        } else if status == "failed" {
            appendSystemStatus(event.message ?? "Turn failed", level: "error")
            statusText = event.message ?? "Turn failed"
        } else {
            if !wasActiveReplySpeechTurn {
                statusText = "Ready"
            }
        }
        if event.isFinal == true {
            isStopping = false
        }
    }

    private func finishAssistantMessages(_ turnID: String) {
        var didChangeConversation = false
        for index in conversation.indices {
            guard case .assistantMessage(var message) = conversation[index], message.turnID == turnID else { continue }
            message.isStreaming = false
            conversation[index] = .assistantMessage(message)
            didChangeConversation = true
        }
        if didChangeConversation {
            noteConversationChanged()
        }
    }

    private func appendRuntimeLog(_ message: String) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        runtimeLogs.append(trimmed)
    }

    private func pendingVoiceDraftCleanup() {
        microphoneAccessTask?.cancel()
        microphoneAccessTask = nil
        cancelMicrophoneRoutePreflight()
        cancelVoiceCaptureWaitingTask()
        clearPendingAssistantDeltaState()
        microphoneRouteCoordinator.leaveRecordingRoute()
        voiceRecordingState = nil
        for turnID in Array(voiceDraftIDsByTurn.keys) {
            removeVoiceDraft(turnID: turnID)
        }
        assistantMessageIDsBySegment.removeAll()
        voiceDraftIDsByTurn.removeAll()
        toolMessageIDsByCall.removeAll()
    }

    private var installablesReadyForChat: Bool {
        modelInstallables.allSatisfy { $0.installState == .installed }
    }

    private func recomputeInstallables() {
        modelInstallables = modelInstallables.map { installable in
            let states = installable.modelIDs.compactMap { underlyingModels[$0] }
            let installState: ModelInstallState
            if states.contains(where: { $0.installState == .failed }) {
                installState = .failed
            } else if states.contains(where: { $0.installState == .downloading }) {
                installState = .downloading
            } else if states.allSatisfy({ $0.installState == .installed }) {
                installState = .installed
            } else {
                installState = .missing
            }

            let warmState: WarmState
            if states.contains(where: { $0.warmState == .error }) {
                warmState = .error
            } else if states.contains(where: { $0.warmState == .warming }) {
                warmState = .warming
            } else if states.allSatisfy({ $0.warmState == .warm }) {
                warmState = .warm
            } else {
                warmState = .cold
            }

            var updated = installable
            updated.installState = installState
            updated.warmState = warmState
            updated.deleteAllowed = installState == .installed
            updated.lastError = states.compactMap(\.lastError).last
            return updated
        }

        let downloadingIDs = Set(modelInstallables.lazy.filter { $0.installState == .downloading }.map(\.id))
        downloadProgressByInstallable = downloadProgressByInstallable.filter { downloadingIDs.contains($0.key) }
    }

    private func appendBootstrapMessage(_ message: String) {
        guard bootstrapMessages.last != message else { return }
        bootstrapMessages.append(message)
        if bootstrapMessages.count > 5 {
            bootstrapMessages.removeFirst(bootstrapMessages.count - 5)
        }
    }

    private func downloadProgressStatusText(for progress: InstallableDownloadProgress, title: String) -> String {
        guard progress.bytesTotal > 0 else {
            return "Downloading \(title)…"
        }

        return "Downloading \(title): \(Self.byteFormatter.string(fromByteCount: progress.bytesDownloaded)) of \(Self.byteFormatter.string(fromByteCount: progress.bytesTotal))"
    }

    private func warmCoreModels() {
        for modelID in [RuntimeModelID.agentModel, .ttsModel] {
            do {
                try runtime.send(RuntimeCommandEnvelope(
                    type: "warm_model",
                    turnID: nil,
                    modelID: modelID.rawValue,
                    installableID: nil,
                    text: nil,
                    speakReply: nil,
                    chunkBase64: nil,
                    sampleRate: nil,
                    channels: nil,
                    toolCallID: nil,
                    attachments: nil,
                    arguments: ["voice_preset": .string(settings.defaultVoicePreset)]
                ))
            } catch {
                appendSystemStatus(error.localizedDescription, level: "error")
            }
        }
    }

    private func warmSTTInBackground() {
        statusText = "Warming voice input…"
        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "warm_model",
                turnID: nil,
                modelID: RuntimeModelID.sttModel.rawValue,
                installableID: nil,
                text: nil,
                speakReply: nil,
                chunkBase64: nil,
                sampleRate: nil,
                channels: nil,
                toolCallID: nil,
                attachments: nil,
                arguments: nil
            ))
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    private func currentWarmState(for modelID: RuntimeModelID) -> WarmState {
        underlyingModels[modelID.rawValue]?.warmState ?? .cold
    }

    private func resolvedComposerText(from trimmedText: String, imageAttachment: ConversationImageAttachment?) -> String {
        if !trimmedText.isEmpty {
            return trimmedText
        }
        return imageAttachment == nil ? "" : Self.defaultImagePrompt
    }

    private func makeConversationImageAttachment(from composerAttachment: ComposerImageAttachment?) -> ConversationImageAttachment? {
        guard let composerAttachment else { return nil }
        return ConversationImageAttachment(
            id: UUID(),
            fileURL: composerAttachment.fileURL,
            displayName: composerAttachment.displayName
        )
    }

    private func runtimeAttachments(from imageAttachment: ConversationImageAttachment?) -> [RuntimeAttachment]? {
        guard let imageAttachment else { return nil }
        return [
            RuntimeAttachment(
                type: "image",
                filePath: imageAttachment.fileURL.path,
                displayName: imageAttachment.displayName
            )
        ]
    }

    private func runtimeHistory(before userMessageIndex: Int) -> [RuntimeHistoryMessage] {
        var orderedTurnIDs: [String] = []
        var userMessagesByTurn: [String: UserMessage] = [:]
        var assistantMessagesByTurn: [String: AssistantMessage] = [:]

        for item in conversation[..<userMessageIndex] {
            switch item {
            case .userMessage(let message):
                if userMessagesByTurn[message.turnID] == nil {
                    orderedTurnIDs.append(message.turnID)
                }
                userMessagesByTurn[message.turnID] = message
            case .assistantMessage(let message):
                if message.isFinalSegment {
                    assistantMessagesByTurn[message.turnID] = message
                }
            default:
                break
            }
        }

        var history: [RuntimeHistoryMessage] = []
        for turnID in orderedTurnIDs {
            guard
                let userMessage = userMessagesByTurn[turnID],
                let assistantMessage = assistantMessagesByTurn[turnID],
                !userMessage.isCancelled,
                !assistantMessage.isCancelled
            else {
                continue
            }

            history.append(RuntimeHistoryMessage(
                role: "user",
                text: userMessage.text,
                attachments: runtimeAttachments(from: userMessage.imageAttachment)
            ))
            history.append(RuntimeHistoryMessage(
                role: "assistant",
                text: assistantMessage.text,
                attachments: nil
            ))
        }
        return history
    }

    private func finalAssistantMessage(for turnID: String) -> AssistantMessage? {
        conversation.compactMap { item -> AssistantMessage? in
            guard case .assistantMessage(let message) = item,
                  message.turnID == turnID,
                  message.isFinalSegment else { return nil }
            return message
        }.last
    }

    private func restorePendingComposerImage(from imageAttachment: ConversationImageAttachment?) {
        guard let imageAttachment else { return }
        pendingComposerImage = ComposerImageAttachment(
            id: UUID(),
            fileURL: imageAttachment.fileURL,
            displayName: imageAttachment.displayName
        )
    }

    private func importImage(from sourceURL: URL) {
        do {
            let attachment = try storeImageAttachment(from: sourceURL, displayName: sourceURL.lastPathComponent)
            pendingComposerImage = attachment
        } catch {
            appendSystemStatus("Failed to import image: \(error.localizedDescription)", level: "error")
        }
    }

    private func importImage(from provider: NSItemProvider) async throws {
        if let fileURL = try await loadDroppedImageFileURL(from: provider) {
            importImage(from: fileURL)
            return
        }

        if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            let image = try await loadDroppedNSImage(from: provider)
            pendingComposerImage = try storeImageAttachment(from: image, displayName: "Dropped Image.png")
            return
        }

        throw NSError(
            domain: "MacAssistant.ImageImport",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Unsupported image drop."]
        )
    }

    private func storeImageAttachment(from sourceURL: URL, displayName: String) throws -> ComposerImageAttachment {
        guard isImageFile(at: sourceURL) else {
            throw NSError(
                domain: "MacAssistant.ImageImport",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "The selected file is not a supported image."]
            )
        }

        try FileManager.default.createDirectory(at: attachmentsDirectoryURL, withIntermediateDirectories: true)
        let fileExtension = normalizedImageExtension(for: sourceURL)
        let destinationURL = attachmentsDirectoryURL
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
            .appendingPathExtension(fileExtension)

        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.copyItem(at: sourceURL, to: destinationURL)

        return ComposerImageAttachment(
            id: UUID(),
            fileURL: destinationURL,
            displayName: displayName.isEmpty ? destinationURL.lastPathComponent : displayName
        )
    }

    private func storeImageAttachment(from image: NSImage, displayName: String) throws -> ComposerImageAttachment {
        try FileManager.default.createDirectory(at: attachmentsDirectoryURL, withIntermediateDirectories: true)
        let destinationURL = attachmentsDirectoryURL
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
            .appendingPathExtension("png")
        guard let pngData = pngData(for: image) else {
            throw NSError(
                domain: "MacAssistant.ImageImport",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "The dropped image could not be converted to PNG."]
            )
        }
        try pngData.write(to: destinationURL, options: .atomic)
        return ComposerImageAttachment(
            id: UUID(),
            fileURL: destinationURL,
            displayName: displayName
        )
    }

    private func isImageFile(at sourceURL: URL) -> Bool {
        if let type = UTType(filenameExtension: sourceURL.pathExtension.lowercased()), type.conforms(to: .image) {
            return true
        }
        return NSImage(contentsOf: sourceURL) != nil
    }

    private func normalizedImageExtension(for sourceURL: URL) -> String {
        let pathExtension = sourceURL.pathExtension.lowercased()
        if let type = UTType(filenameExtension: pathExtension), type.conforms(to: .image), !pathExtension.isEmpty {
            return pathExtension
        }
        return "png"
    }

    private func loadDroppedImageFileURL(from provider: NSItemProvider) async throws -> URL? {
        if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            return try await withCheckedThrowingContinuation { continuation in
                provider.loadFileRepresentation(forTypeIdentifier: UTType.image.identifier) { url, error in
                    if let error {
                        continuation.resume(throwing: error)
                        return
                    }
                    guard let url else {
                        continuation.resume(returning: nil)
                        return
                    }
                    do {
                        let temporaryURL = FileManager.default.temporaryDirectory
                            .appendingPathComponent(UUID().uuidString, isDirectory: false)
                            .appendingPathExtension(url.pathExtension.isEmpty ? "png" : url.pathExtension)
                        if FileManager.default.fileExists(atPath: temporaryURL.path) {
                            try FileManager.default.removeItem(at: temporaryURL)
                        }
                        try FileManager.default.copyItem(at: url, to: temporaryURL)
                        continuation.resume(returning: temporaryURL)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        }

        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            return try await withCheckedThrowingContinuation { continuation in
                provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, error in
                    if let error {
                        continuation.resume(throwing: error)
                        return
                    }
                    if let url = item as? URL {
                        continuation.resume(returning: url)
                        return
                    }
                    if let nsurl = item as? NSURL, let url = nsurl as URL? {
                        continuation.resume(returning: url)
                        return
                    }
                    if let data = item as? Data, let url = URL(dataRepresentation: data, relativeTo: nil) {
                        continuation.resume(returning: url)
                        return
                    }
                    continuation.resume(returning: nil)
                }
            }
        }

        return nil
    }

    private func loadDroppedNSImage(from provider: NSItemProvider) async throws -> NSImage {
        try await withCheckedThrowingContinuation { continuation in
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard
                    let data,
                    let image = NSImage(data: data)
                else {
                    continuation.resume(throwing: NSError(
                        domain: "MacAssistant.ImageImport",
                        code: 4,
                        userInfo: [NSLocalizedDescriptionKey: "No image data was dropped."]
                    ))
                    return
                }
                continuation.resume(returning: image)
            }
        }
    }

    private func pngData(for image: NSImage) -> Data? {
        guard
            let tiffData = image.tiffRepresentation,
            let bitmap = NSBitmapImageRep(data: tiffData)
        else {
            return nil
        }
        return bitmap.representation(using: .png, properties: [:])
    }

    private static func supportsImageDrop(_ provider: NSItemProvider) -> Bool {
        provider.hasItemConformingToTypeIdentifier(UTType.image.identifier)
            || provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier)
    }

    private func noteConversationChanged() {
        transcriptRevision &+= 1
    }

    private func scheduleAssistantDeltaFlushIfNeeded() {
        guard assistantDeltaFlushTask == nil else { return }
        let delay = assistantDeltaFlushDelay
        assistantDeltaFlushTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.flushPendingAssistantDeltas()
        }
    }

    private func clearPendingAssistantDeltaState() {
        assistantDeltaFlushTask?.cancel()
        assistantDeltaFlushTask = nil
        pendingAssistantDeltasBySegment.removeAll()
    }

    private func enqueueAssistantDelta(
        turnID: String,
        segmentID: String,
        delta: String,
        isFinalSegment: Bool?
    ) {
        if assistantMessageIDsBySegment[segmentID] == nil {
            let message = AssistantMessage(
                id: UUID(),
                turnID: turnID,
                segmentID: segmentID,
                text: delta,
                isStreaming: isFinalSegment != true,
                isCancelled: false,
                isFinalSegment: isFinalSegment ?? false,
                source: voiceDraftIDsByTurn[turnID] == nil ? .typed : .voice
            )
            let index = conversation.count
            conversation.append(.assistantMessage(message))
            assistantMessageIDsBySegment[segmentID] = index
            noteConversationChanged()
            return
        }

        var pendingDelta = pendingAssistantDeltasBySegment[segmentID] ?? PendingAssistantDelta()
        pendingDelta.text += delta
        if let isFinalSegment {
            pendingDelta.isFinalSegment = isFinalSegment
        }
        pendingAssistantDeltasBySegment[segmentID] = pendingDelta

        if pendingDelta.isFinalSegment == true {
            flushPendingAssistantDelta(segmentID)
        } else {
            scheduleAssistantDeltaFlushIfNeeded()
        }
    }

    private func flushPendingAssistantDeltas() {
        assistantDeltaFlushTask?.cancel()
        assistantDeltaFlushTask = nil

        let orderedSegmentIDs = pendingAssistantDeltasBySegment.keys.sorted {
            (assistantMessageIDsBySegment[$0] ?? Int.max) < (assistantMessageIDsBySegment[$1] ?? Int.max)
        }
        for segmentID in orderedSegmentIDs {
            flushPendingAssistantDelta(segmentID)
        }
    }

    private func flushPendingAssistantDelta(_ segmentID: String) {
        guard let pendingDelta = pendingAssistantDeltasBySegment.removeValue(forKey: segmentID),
              let index = assistantMessageIDsBySegment[segmentID],
              case .assistantMessage(var message) = conversation[index] else { return }

        if !pendingDelta.text.isEmpty {
            message.text += pendingDelta.text
        }
        if let isFinalSegment = pendingDelta.isFinalSegment {
            message.isFinalSegment = isFinalSegment
            message.isStreaming = !isFinalSegment
        } else {
            message.isStreaming = true
        }
        conversation[index] = .assistantMessage(message)
        noteConversationChanged()

        if pendingAssistantDeltasBySegment.isEmpty {
            assistantDeltaFlushTask?.cancel()
            assistantDeltaFlushTask = nil
        }
    }

    private func updateVoiceDraft(turnID: String, text: String) {
        guard let index = voiceDraftIDsByTurn[turnID] else { return }
        if case .userVoiceDraft(var draft) = conversation[index] {
            draft.text = text
            conversation[index] = .userVoiceDraft(draft)
            noteConversationChanged()
        }
    }

    @discardableResult
    private func commitVoiceDraft(turnID: String, text: String) -> Bool {
        guard let index = voiceDraftIDsByTurn[turnID],
              case .userVoiceDraft(let draft) = conversation[index] else { return false }
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedText.isEmpty, draft.imageAttachment == nil {
            conversation.remove(at: index)
            voiceDraftIDsByTurn.removeValue(forKey: turnID)
            rebuildConversationIndices()
            noteConversationChanged()
            return false
        }
        let resolvedText = resolvedComposerText(from: trimmedText, imageAttachment: draft.imageAttachment)
        conversation[index] = .userMessage(UserMessage(
            id: draft.id,
            turnID: turnID,
            text: resolvedText,
            imageAttachment: draft.imageAttachment,
            source: .voice,
            isCancelled: false
        ))
        voiceDraftIDsByTurn.removeValue(forKey: turnID)
        noteConversationChanged()
        return true
    }

    private func removeVoiceDraft(turnID: String, clearActiveResponseTurn: Bool = true) {
        guard let index = voiceDraftIDsByTurn.removeValue(forKey: turnID) else { return }
        conversation.remove(at: index)
        rebuildConversationIndices()
        noteConversationChanged()
        activeTurnIDs.remove(turnID)
        if clearActiveResponseTurn, activeResponseTurnID == turnID {
            activeResponseTurnID = nil
        }
    }

    private func rerunTurn(turnID: String, updatedText: String?, updatedSource: TurnSource?) {
        guard let userIndex = conversation.firstIndex(where: {
            guard case .userMessage(let message) = $0 else { return false }
            return message.turnID == turnID
        }) else {
            return
        }

        guard case .userMessage(var userMessage) = conversation[userIndex] else { return }

        cancelActiveSpeechPlaybackIfNeeded()

        if let updatedText {
            let trimmedText = updatedText.trimmingCharacters(in: .whitespacesAndNewlines)
            userMessage.text = resolvedComposerText(
                from: trimmedText,
                imageAttachment: userMessage.imageAttachment
            )
        }
        if let updatedSource {
            userMessage.source = updatedSource
        }
        userMessage.isCancelled = false

        let prefixHistory = runtimeHistory(before: userIndex)
        let trailingItems: ArraySlice<ConversationItem> = conversation.indices.contains(userIndex + 1)
            ? conversation[(userIndex + 1)...]
            : ArraySlice()
        let removedTurnIDs = Set(trailingItems.compactMap(\.turnID))

        conversation[userIndex] = .userMessage(userMessage)
        conversation = Array(conversation.prefix(userIndex + 1))
        discardedTurnIDs.formUnion(removedTurnIDs)
        discardedTurnIDs.remove(turnID)
        activeTurnIDs.removeAll()
        activeTurnIDs.insert(turnID)
        activeResponseTurnID = turnID
        isStopping = false
        statusText = "Thinking…"
        clearPendingAssistantDeltaState()
        rebuildConversationIndices()
        noteConversationChanged()

        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "replace_history",
                history: prefixHistory
            ))
            try sendTextTurn(
                turnID: turnID,
                text: userMessage.text,
                imageAttachment: userMessage.imageAttachment,
                speakReply: false
            )
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
    }

    private func rebuildConversationIndices() {
        assistantMessageIDsBySegment.removeAll()
        voiceDraftIDsByTurn.removeAll()
        toolMessageIDsByCall.removeAll()

        for (index, item) in conversation.enumerated() {
            switch item {
            case .userVoiceDraft(let draft):
                voiceDraftIDsByTurn[draft.turnID] = index
            case .assistantMessage(let message):
                assistantMessageIDsBySegment[message.segmentID] = index
            case .toolInvocation(let tool):
                toolMessageIDsByCall[tool.toolCallID] = index
            default:
                break
            }
        }
    }

    private func updateTool(_ toolCallID: String, mutate: (inout ToolInvocation) -> Void) {
        guard let index = toolMessageIDsByCall[toolCallID],
              case .toolInvocation(var tool) = conversation[index] else { return }
        mutate(&tool)
        conversation[index] = .toolInvocation(tool)
        noteConversationChanged()
    }

    private func markTurnCancelled(_ turnID: String) {
        var didChangeConversation = false
        for index in conversation.indices {
            switch conversation[index] {
            case .userVoiceDraft(var draft) where draft.turnID == turnID:
                draft.isCancelled = true
                conversation[index] = .userVoiceDraft(draft)
                didChangeConversation = true
            case .userMessage(var message) where message.turnID == turnID:
                message.isCancelled = true
                conversation[index] = .userMessage(message)
                didChangeConversation = true
            case .assistantMessage(var message) where message.turnID == turnID:
                message.isCancelled = true
                message.isStreaming = false
                conversation[index] = .assistantMessage(message)
                didChangeConversation = true
            case .toolInvocation(var tool) where tool.turnID == turnID:
                tool.executionState = .cancelled
                conversation[index] = .toolInvocation(tool)
                didChangeConversation = true
            default:
                break
            }
        }
        if didChangeConversation {
            noteConversationChanged()
        }
    }

    private func appendSystemStatus(_ text: String, level: String) {
        if case .systemStatus(let lastStatus)? = conversation.last,
           lastStatus.text == text,
           lastStatus.level == level {
            return
        }
        conversation.append(.systemStatus(SystemStatusMessage(id: UUID(), turnID: nil, text: text, level: level)))
        noteConversationChanged()
    }

    private func sendTextTurn(
        turnID: String,
        text: String,
        imageAttachment: ConversationImageAttachment?,
        speakReply: Bool
    ) throws {
        try runtime.send(RuntimeCommandEnvelope(
            type: "send_text",
            turnID: turnID,
            text: text,
            speakReply: speakReply,
            attachments: runtimeAttachments(from: imageAttachment),
            arguments: ["voice_preset": .string(settings.defaultVoicePreset)]
        ))
    }

    private func cancelActiveSpeechPlaybackIfNeeded() {
        playbackService.stop()
        let turnID = activeSpeechTurnID
        let hadTranscriptSpeech = activeSpeechSourceAssistantSegmentID != nil
        resetTranscriptSpeechState()
        guard let turnID else {
            if hadTranscriptSpeech {
                clearReadyStatusAfterSpeechIfNeeded()
            }
            return
        }
        activeTurnIDs.remove(turnID)
        speechOnlyTurnIDs.remove(turnID)
        do {
            try runtime.send(RuntimeCommandEnvelope(
                type: "cancel",
                turnID: turnID
            ))
        } catch {
            appendSystemStatus(error.localizedDescription, level: "error")
        }
        clearReadyStatusAfterSpeechIfNeeded()
    }

    private func handlePlaybackStarted() {
        if activeSpeechSourceAssistantSegmentID != nil {
            transcriptSpeechPhase = .playing
            return
        }
        guard activeReplySpeechTurnID != nil else { return }
        replySpeechPhase = .playing
        statusText = "Speaking…"
    }

    private func handlePlaybackFinished() {
        if activeSpeechSourceAssistantSegmentID != nil {
            if activeSpeechTurnID != nil {
                transcriptSpeechPhase = .synthesizing
                return
            }
            resetTranscriptSpeechState()
            clearReadyStatusAfterSpeechIfNeeded()
            return
        }
        guard activeReplySpeechTurnID != nil else { return }
        resetReplySpeechState()
        clearReadyStatusAfterSpeechIfNeeded()
    }

    private func resetTranscriptSpeechState() {
        activeSpeechTurnID = nil
        activeSpeechSourceAssistantSegmentID = nil
        transcriptSpeechPhase = .idle
    }

    private func beginReplySpeechIfNeeded(for turnID: String) {
        guard finalAssistantMessage(for: turnID) != nil else { return }
        guard activeReplySpeechTurnID != turnID else { return }
        activeReplySpeechTurnID = turnID
        replySpeechPhase = .buffering
        statusText = "Speaking…"
    }

    private func resetReplySpeechState(turnID: String? = nil) {
        guard turnID == nil || activeReplySpeechTurnID == turnID else { return }
        activeReplySpeechTurnID = nil
        replySpeechPhase = .idle
    }

    private func clearReadyStatusAfterSpeechIfNeeded() {
        guard !hasLiveVoiceDraft, !canStopCurrentOutput else { return }
        statusText = "Ready"
    }

    private static func describe(duration: Duration) -> String {
        let components = duration.components
        let seconds = Double(components.seconds) + Double(components.attoseconds) / 1_000_000_000_000_000_000
        if seconds >= 1 {
            return String(format: "%.1fs", seconds)
        }
        return String(format: "%.0fms", seconds * 1000)
    }

    private func copyTextToPasteboard(_ text: String) {
        guard !text.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }
}
