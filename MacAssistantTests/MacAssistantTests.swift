import AVFoundation
import AppKit
import Foundation
import Testing
@testable import MacAssistant

struct MacAssistantTests {
    @Test
    @MainActor
    func runtimeReadyAfterMissingModelsShowsSetupInsteadOfBootstrapping() {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "bootstrap_progress",
            message: "No local models found yet.",
            stage: "Checking installed models"
        ))
        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.agentModel.rawValue,
            installState: ModelInstallState.missing.rawValue,
            warmState: WarmState.cold.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.ttsModel.rawValue,
            installState: ModelInstallState.missing.rawValue,
            warmState: WarmState.cold.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.sttModel.rawValue,
            installState: ModelInstallState.missing.rawValue,
            warmState: WarmState.cold.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "bootstrap_progress",
            message: "Runtime connected. Waiting for model downloads.",
            stage: "Runtime ready"
        ))

        #expect(model.phase == .setup)
        #expect(model.statusText == "Runtime connected. Waiting for model downloads.")
    }

    @Test
    @MainActor
    func coreModelWarmFailureReturnsToSetup() {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.agentModel.rawValue,
            installState: ModelInstallState.installed.rawValue,
            warmState: WarmState.error.rawValue,
            message: "agent failed to load"
        ))
        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.ttsModel.rawValue,
            installState: ModelInstallState.installed.rawValue,
            warmState: WarmState.cold.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.sttModel.rawValue,
            installState: ModelInstallState.installed.rawValue,
            warmState: WarmState.cold.rawValue
        ))

        #expect(model.phase == .setup)
        #expect(model.statusText == "A required model failed to load.")
    }

    @Test
    func installableAggregationPrefersInstalledWhenAllModelsPresent() {
        var installable = ModelInstallable.defaults[0]
        let states = [
            UnderlyingModelState(id: "stt_model", installState: .installed, warmState: .warm, lastError: nil),
            UnderlyingModelState(id: "tts_model", installState: .installed, warmState: .cold, lastError: nil)
        ]

        installable.installState = states.allSatisfy { $0.installState == .installed } ? .installed : .missing
        installable.warmState = states.allSatisfy { $0.warmState == .warm } ? .warm : .cold

        #expect(installable.installState == .installed)
        #expect(installable.warmState == .cold)
    }

    @Test
    func defaultInstallablesUseGenericModelNames() {
        #expect(ModelInstallable.defaults.map(\.id) == ["voice_pack", "agent_model"])
        #expect(ModelInstallable.defaults[0].modelIDs == ["stt_model", "tts_model"])
        #expect(ModelInstallable.defaults[1].modelIDs == ["agent_model"])
    }

    @Test
    @MainActor
    func downloadProgressEventsUpdateInstallableProgress() {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "download_progress",
            modelID: RuntimeModelID.sttModel.rawValue,
            installableID: "voice_pack",
            bytesDownloaded: 512,
            bytesTotal: 1024,
            etaSeconds: 2.0,
            speedBytesPerSecond: 256
        ))

        let progress = model.downloadProgress(for: "voice_pack")
        #expect(progress?.bytesDownloaded == 512)
        #expect(progress?.bytesTotal == 1024)
        #expect(progress?.fractionCompleted == 0.5)
    }

    @Test
    func toolSafetyStatesRoundTrip() {
        let tool = ToolInvocation(
            id: UUID(),
            turnID: "turn-1",
            toolCallID: "tool-1",
            name: "execute_script",
            arguments: ["kb_script_id": .string("finder_open_downloads")],
            safetyClass: .mutating,
            approvalState: .pending,
            executionState: .proposed,
            summary: "Open Downloads in Finder",
            output: "",
            isExpanded: false
        )

        #expect(tool.safetyClass == SafetyClass.mutating)
        #expect(tool.approvalState == ApprovalState.pending)
        #expect(tool.executionState == ExecutionState.proposed)
    }

    @Test
    @MainActor
    func floatingWindowConfigurationUsesCrossApplicationAllSpacesBehavior() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 300),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.level = .floating
        window.collectionBehavior = [.moveToActiveSpace, .fullScreenPrimary]

        FloatingAssistantWindowConfiguration.apply(to: window)

        #expect(window.level == .floating)
        #expect(window.collectionBehavior.contains(.canJoinAllSpaces))
        #expect(window.collectionBehavior.contains(.canJoinAllApplications))
        #expect(window.collectionBehavior.contains(.fullScreenAuxiliary))
        #expect(!window.collectionBehavior.contains(.moveToActiveSpace))
        #expect(!window.collectionBehavior.contains(.fullScreenPrimary))
    }

    @Test
    @MainActor
    func completedTurnClearsAssistantStreamingState() {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: "turn-1",
            text: "Hello there"
        ))
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: "turn-1",
            status: "finished",
            isFinal: true
        ))

        guard case .assistantMessage(let message)? = model.conversation.last else {
            Issue.record("Expected final conversation item to be an assistant message.")
            return
        }

        #expect(message.text == "Hello there")
        #expect(message.isStreaming == false)
    }

    @Test
    func suppressesBenignRuntimeNoise() {
        #expect(AgentRuntimeClient.shouldSuppressRuntimeLog("Loaded Mistral tokenizer from /tmp/tekken.json"))
        #expect(AgentRuntimeClient.shouldSuppressRuntimeLog("Resolving dependencies"))
        #expect(AgentRuntimeClient.shouldSuppressRuntimeLog("[2026-03-29T03:46:56.999Z] [macos_automator_server] [INFO] Starting macos_automator MCP Server..."))
        #expect(!AgentRuntimeClient.shouldSuppressRuntimeLog("stderr: Fatal error in server"))
    }

    @Test
    @MainActor
    func duplicateRuntimeErrorsDoNotSpamSystemStatus() {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "error",
            message: "'str' object has no attribute 'get'"
        ))
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: "turn-1",
            message: "'str' object has no attribute 'get'",
            status: "failed",
            isFinal: true
        ))

        let statuses = model.conversation.compactMap { item -> SystemStatusMessage? in
            guard case .systemStatus(let status) = item else { return nil }
            return status
        }

        #expect(statuses.count == 1)
        #expect(statuses.first?.text == "'str' object has no attribute 'get'")
    }

    @Test
    func microphoneConvertedFrameCapacityTracksRealTimeDuration() {
        let inputFrames: AVAudioFrameCount = 2_048
        let outputCapacity = MicrophoneCaptureService.convertedFrameCapacity(
            inputFrameLength: inputFrames,
            inputSampleRate: 44_100,
            outputSampleRate: 16_000
        )
        let convertedFrames = Int(outputCapacity) - 1
        let inputDuration = Double(inputFrames) / 44_100
        let outputDuration = Double(convertedFrames) / 16_000

        #expect(abs(inputDuration - outputDuration) < 0.002)
    }

    @Test
    func microphoneInputBlockConsumesBufferOnlyOnce() throws {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 44_100,
            channels: 1,
            interleaved: false
        )!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16)!
        buffer.frameLength = 16
        let inputBlock = MicrophoneCaptureService.makeSingleUseInputBlock(buffer: buffer)

        var firstStatus = AVAudioConverterInputStatus.noDataNow
        let firstBuffer = withUnsafeMutablePointer(to: &firstStatus) { pointer in
            inputBlock(0, pointer)
        }

        var secondStatus = AVAudioConverterInputStatus.haveData
        let secondBuffer = withUnsafeMutablePointer(to: &secondStatus) { pointer in
            inputBlock(0, pointer)
        }

        #expect(firstStatus == .haveData)
        #expect(firstBuffer === buffer)
        #expect(secondStatus == .noDataNow)
        #expect(secondBuffer == nil)
    }

    @Test
    @MainActor
    func pendingApprovalToolStartsExpanded() {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "tool_proposed",
            turnID: "turn-1",
            toolCallID: "tool-1",
            toolName: "execute_script",
            toolSummary: "execute_script • pending",
            toolArguments: ["script_content": .string("return system info")],
            approvalState: ApprovalState.pending.rawValue,
            executionState: ExecutionState.proposed.rawValue
        ))

        guard case .toolInvocation(let tool)? = model.conversation.last else {
            Issue.record("Expected final conversation item to be a tool invocation.")
            return
        }

        #expect(tool.isExpanded)
        #expect(tool.approvalState == .pending)
    }

    @Test
    @MainActor
    func alwaysAcceptToolCallsAutoApprovesPendingTools() {
        let model = AppModel()
        model.settings.alwaysAcceptToolCalls = true

        model.handle(event: runtimeEvent(
            type: "tool_proposed",
            turnID: "turn-1",
            toolCallID: "tool-1",
            toolName: "execute_script",
            toolSummary: "execute_script • pending",
            toolArguments: ["script_content": .string("tell application \"Mail\" to activate")],
            approvalState: ApprovalState.pending.rawValue,
            executionState: ExecutionState.proposed.rawValue
        ))

        guard case .toolInvocation(let tool)? = model.conversation.last else {
            Issue.record("Expected final conversation item to be a tool invocation.")
            return
        }

        #expect(tool.approvalState == .approved)
        #expect(tool.executionState == .running)
    }

    @Test
    @MainActor
    func settingsPersistAcrossModelSessions() throws {
        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: tempDirectory)
        }

        let firstModel = AppModel(appSupportURL: tempDirectory, persistSettings: true)
        firstModel.settings.defaultVoicePreset = "neutral_female"
        firstModel.settings.inputDevicePreference = .specificDeviceUID("built-in-mic")
        firstModel.settings.launchOnOpen = true
        firstModel.settings.alwaysAcceptToolCalls = true

        let secondModel = AppModel(appSupportURL: tempDirectory, persistSettings: true)

        #expect(secondModel.settings.defaultVoicePreset == "neutral_female")
        #expect(secondModel.settings.inputDevicePreference == .specificDeviceUID("built-in-mic"))
        #expect(secondModel.settings.launchOnOpen)
        #expect(secondModel.settings.alwaysAcceptToolCalls)
    }

    @Test
    @MainActor
    func resetConversationClearsChatAndIgnoresLateTurnUpdates() {
        let model = AppModel()

        model.composerText = "draft"
        model.sendTextMessage()
        let turnID = model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last?.turnID

        model.resetConversation()
        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: turnID,
            text: "stale output"
        ))
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: turnID,
            status: "cancelled",
            isFinal: true
        ))

        #expect(model.conversation.isEmpty)
        #expect(model.composerText.isEmpty)
        #expect(model.statusText == "Ready")
    }

    @Test
    @MainActor
    func attachmentOnlySendUsesDefaultPromptAndDispatchesImageAttachment() throws {
        let model = AppModel()
        let imageURL = try makeTestImageURL()
        defer { try? FileManager.default.removeItem(at: imageURL.deletingLastPathComponent()) }

        model.pendingComposerImage = ComposerImageAttachment(
            id: UUID(),
            fileURL: imageURL,
            displayName: "sample.png"
        )

        model.sendTextMessage()

        let userMessages = model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }

        #expect(userMessages.last?.text == AppModel.defaultImagePrompt)
        #expect(userMessages.last?.imageAttachment?.fileURL == imageURL)
        #expect(model.pendingComposerImage == nil)
        #expect(model.lastRuntimeCommand?.text == AppModel.defaultImagePrompt)
        #expect(model.lastRuntimeCommand?.attachments?.first?.filePath == imageURL.path)
    }

    @Test
    @MainActor
    func secondTypedSendIsIgnoredWhileTurnIsActive() {
        let model = AppModel()
        model.composerText = "First request"

        model.sendTextMessage()
        let firstCommand = model.lastRuntimeCommand

        model.composerText = "Second request"
        model.sendTextMessage()

        let userMessages = model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }

        #expect(userMessages.count == 1)
        #expect(userMessages.last?.text == "First request")
        #expect(model.composerText == "Second request")
        #expect(model.lastRuntimeCommand == firstCommand)
        #expect(model.isAwaitingAgentResponse)
        #expect(model.isComposerLocked)
    }

    @Test
    @MainActor
    func typedSendWithAttachmentKeepsImageOnConversationTurn() throws {
        let model = AppModel()
        let imageURL = try makeTestImageURL()
        defer { try? FileManager.default.removeItem(at: imageURL.deletingLastPathComponent()) }

        model.pendingComposerImage = ComposerImageAttachment(
            id: UUID(),
            fileURL: imageURL,
            displayName: "sample.png"
        )
        model.composerText = "What is in this image?"

        model.sendTextMessage()

        let userMessages = model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }

        #expect(userMessages.last?.text == "What is in this image?")
        #expect(userMessages.last?.imageAttachment?.displayName == "sample.png")
        #expect(model.lastRuntimeCommand?.attachments?.first?.displayName == "sample.png")
    }

    @Test
    @MainActor
    func micStartAndImageDropAreIgnoredWhileTurnIsActive() throws {
        let model = AppModel()
        model.isMicReady = true
        model.composerText = "Active request"
        model.sendTextMessage()

        let imageURL = try makeTestImageURL()
        defer { try? FileManager.default.removeItem(at: imageURL.deletingLastPathComponent()) }

        let acceptedDrop = model.importDroppedImage(from: [try #require(NSItemProvider(contentsOf: imageURL))])
        model.startRecording()

        let voiceDrafts = model.conversation.compactMap { item -> UserVoiceDraft? in
            guard case .userVoiceDraft(let draft) = item else { return nil }
            return draft
        }

        #expect(acceptedDrop == false)
        #expect(model.pendingComposerImage == nil)
        #expect(model.isRecording == false)
        #expect(voiceDrafts.isEmpty)
    }

    @Test
    @MainActor
    func startingVoiceTurnCreatesDraftBubbleAndWaitsForFirstChunkBeforeOpeningRuntime() {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true
        model.composerText = "Typed draft"

        model.startRecording()

        guard case .userVoiceDraft(let draft)? = model.conversation.last else {
            Issue.record("Expected a voice draft to be added to the conversation.")
            return
        }

        #expect(draft.text.isEmpty)
        #expect(draft.source == .voice)
        #expect(model.lastRuntimeCommand == nil)
        #expect(model.composerText == "Typed draft")
        #expect(microphoneService.startCallCount == 1)
        #expect(model.hasLiveVoiceDraft)
    }

    @Test
    @MainActor
    func delayedFirstBufferTransitionsVoiceTurnToWaitingStateInsteadOfFailing() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            voiceCaptureWaitingDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        #expect(currentVoiceDraft(in: model) != nil)
        #expect(model.isRecording)
        #expect(model.isWaitingForMicrophoneAudio)
        #expect(model.statusText == "Waiting for microphone audio…")
        #expect(systemStatuses(in: model).isEmpty)
        #expect(runtime.sentCommands.isEmpty)
    }

    @Test
    @MainActor
    func lateFirstChunkClearsWaitingStateAndStartsRuntimeStreaming() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            voiceCaptureWaitingDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()
        #expect(model.isWaitingForMicrophoneAudio)

        microphoneService.emitInputBuffer()
        await settleVoiceCallbacks()

        #expect(model.isWaitingForMicrophoneAudio == false)
        #expect(model.statusText == "Listening…")
        #expect(runtime.sentCommands.isEmpty)

        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        #expect(runtime.sentCommands.map(\.type) == ["start_recording", "append_audio_chunk"])
        #expect(model.lastRuntimeCommand?.type == "append_audio_chunk")
    }

    @Test
    @MainActor
    func transcriptDeltaUpdatesLiveVoiceDraftCumulatively() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        model.handle(event: runtimeEvent(type: "transcript_delta", turnID: turnID, text: "Hello"))
        model.handle(event: runtimeEvent(type: "transcript_delta", turnID: turnID, text: "Hello there"))

        #expect(currentVoiceDraft(in: model)?.text == "Hello there")
        #expect(model.lastRuntimeCommand?.type == "append_audio_chunk")
        #expect(model.lastRuntimeCommand?.turnID == turnID)
    }

    @Test
    @MainActor
    func sendDuringRecordingFinalizesVoiceTurnWithoutFallingBackToTextSend() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        model.sendCurrentInput()

        #expect(model.lastRuntimeCommand?.type == "stop_recording")
        #expect(model.lastRuntimeCommand?.turnID == turnID)
        #expect(model.isRecording == false)
        #expect(model.hasLiveVoiceDraft)

        model.handle(event: runtimeEvent(type: "transcript_final", turnID: turnID, text: "voice input"))

        guard case .userMessage(let message)? = model.conversation.last else {
            Issue.record("Expected the voice draft to commit into a user message.")
            return
        }

        #expect(message.turnID == turnID)
        #expect(message.text == "voice input")
        #expect(message.source == .voice)
        #expect(model.lastRuntimeCommand?.type == "stop_recording")
    }

    @Test
    @MainActor
    func sendWhileWaitingForAudioRemovesDraftAndShowsExplicitNoAudioError() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            voiceCaptureWaitingDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        model.sendCurrentInput()

        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: turnID, in: model) == nil)
        #expect(model.isRecording == false)
        #expect(model.statusText == "No microphone audio was captured. Try speaking again.")
        #expect(runtime.sentCommands.isEmpty)
        #expect(systemStatuses(in: model).map(\.text) == ["No microphone audio was captured. Try speaking again."])
    }

    @Test
    @MainActor
    func micStopDiscardsVoiceDraftAndLeavesMicReusable() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        let firstTurnID = try #require(currentVoiceDraft(in: model)?.turnID)
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        model.discardCurrentInput()

        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: firstTurnID, in: model) == nil)
        #expect(model.lastRuntimeCommand?.type == "cancel")
        #expect(model.lastRuntimeCommand?.turnID == firstTurnID)
        #expect(model.isRecording == false)
        #expect(model.hasLiveVoiceDraft == false)
        #expect(microphoneService.stopCallCount >= 1)

        model.startRecording()

        #expect(microphoneService.startCallCount == 2)
        #expect(currentVoiceDraft(in: model)?.turnID != firstTurnID)
    }

    @Test
    @MainActor
    func micStopWhileWaitingDiscardsDraftSilentlyAndLeavesMicReusable() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            voiceCaptureWaitingDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        let firstTurnID = try #require(currentVoiceDraft(in: model)?.turnID)
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        model.discardCurrentInput()

        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: firstTurnID, in: model) == nil)
        #expect(systemStatuses(in: model).isEmpty)
        #expect(runtime.sentCommands.isEmpty)
        #expect(model.statusText == "Ready")
        #expect(model.isRecording == false)

        model.startRecording()

        #expect(microphoneService.startCallCount == 2)
        #expect(currentVoiceDraft(in: model)?.turnID != firstTurnID)
    }

    @Test
    @MainActor
    func sendDuringSilentImageVoiceTurnFallsBackToImagePrompt() throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true
        let imageURL = try makeTestImageURL()
        defer { try? FileManager.default.removeItem(at: imageURL.deletingLastPathComponent()) }

        model.pendingComposerImage = ComposerImageAttachment(
            id: UUID(),
            fileURL: imageURL,
            displayName: "sample.png"
        )

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)

        model.sendCurrentInput()

        let message = try #require(userMessage(for: turnID, in: model))
        #expect(message.text == AppModel.defaultImagePrompt)
        #expect(message.imageAttachment?.fileURL == imageURL)
        #expect(message.source == .voice)
        #expect(model.lastRuntimeCommand?.type == "send_text")
        #expect(model.lastRuntimeCommand?.attachments?.first?.filePath == imageURL.path)
    }

    @Test
    @MainActor
    func emptyTranscriptFinalAndFailedTurnFinishedRemoveVoiceDraft() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        microphoneService.emitChunk()
        await settleVoiceCallbacks()
        model.sendCurrentInput()

        model.handle(event: runtimeEvent(type: "transcript_final", turnID: turnID, text: ""))
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: turnID,
            message: "No text captured.",
            status: "failed",
            isFinal: true
        ))

        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: turnID, in: model) == nil)
        #expect(model.statusText == "No text captured.")
    }

    @Test
    @MainActor
    func failedVoiceTurnLeavesMicImmediatelyReusable() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        let firstTurnID = try #require(currentVoiceDraft(in: model)?.turnID)
        microphoneService.emitError(NSError(
            domain: "MacAssistantTests.Microphone",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Capture failed."]
        ))
        await settleVoiceCallbacks()

        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: firstTurnID, in: model) == nil)
        #expect(model.statusText == "Capture failed.")
        #expect(model.isRecording == false)

        model.startRecording()

        #expect(microphoneService.startCallCount == 2)
        #expect(currentVoiceDraft(in: model)?.turnID != firstTurnID)
    }

    @Test
    @MainActor
    func retryableCaptureStartFailureRetriesSilentlyAndKeepsVoiceDraft() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let retryableError = NSError(
            domain: NSOSStatusErrorDomain,
            code: -10868,
            userInfo: [NSLocalizedDescriptionKey: "Format not supported."]
        )
        microphoneService.startResults = [.failure(retryableError), .success(())]
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneStartRetryDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)

        #expect(microphoneService.startCallCount == 1)
        #expect(currentVoiceDraft(in: model)?.turnID == turnID)
        #expect(systemStatuses(in: model).isEmpty)

        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        #expect(microphoneService.startCallCount == 2)
        #expect(currentVoiceDraft(in: model)?.turnID == turnID)
        #expect(model.isRecording)
        #expect(systemStatuses(in: model).isEmpty)
        #expect(model.statusText == "Listening…")
    }

    @Test
    @MainActor
    func repeatedRetryableCaptureStartFailureSurfacesErrorOnce() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let retryableError = NSError(
            domain: NSOSStatusErrorDomain,
            code: -10868,
            userInfo: [NSLocalizedDescriptionKey: "Format not supported."]
        )
        microphoneService.startResults = [.failure(retryableError), .failure(retryableError)]
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneStartRetryDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)

        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        #expect(microphoneService.startCallCount == 2)
        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: turnID, in: model) == nil)
        #expect(model.statusText == "Format not supported.")
        #expect(systemStatuses(in: model).map(\.text) == ["Format not supported."])
    }

    @Test
    @MainActor
    func discardingVoiceDraftCancelsPendingCaptureStartRetry() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let retryableError = NSError(
            domain: NSOSStatusErrorDomain,
            code: -10868,
            userInfo: [NSLocalizedDescriptionKey: "Format not supported."]
        )
        microphoneService.startResults = [.failure(retryableError), .success(())]
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneStartRetryDelay: .milliseconds(50)
        )
        model.isMicReady = true

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        #expect(microphoneService.startCallCount == 1)

        model.discardCurrentInput()
        try? await Task.sleep(for: .milliseconds(80))
        await settleVoiceCallbacks()

        #expect(microphoneService.startCallCount == 1)
        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: turnID, in: model) == nil)
        #expect(systemStatuses(in: model).isEmpty)
        #expect(model.statusText == "Ready")
    }

    @Test
    @MainActor
    func microphoneDiagnosticsAppearInDebugLogs() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            voiceCaptureWaitingDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()
        microphoneService.emitInputBuffer()
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        #expect(model.runtimeLogsText.contains("[Mic] Engine started"))
        #expect(model.runtimeLogsText.contains("[Mic] Waiting for microphone audio"))
        #expect(model.runtimeLogsText.contains("[Mic] First input buffer"))
        #expect(model.runtimeLogsText.contains("[Mic] First converted chunk"))
    }

    @Test
    @MainActor
    func automaticInputDevicePreferencePrefersBuiltInMicrophoneOverAirPods() {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let airPodsMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 202,
            uid: "airpods-mic",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        microphoneService.inputDevices = [builtInMicrophone, airPodsMicrophone]
        microphoneService.defaultInputDeviceUID = airPodsMicrophone.uid

        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true
        model.refreshAvailableInputDevices()

        model.startRecording()

        #expect(microphoneService.lastPreferredInputDeviceUID == builtInMicrophone.uid)
        #expect(model.runtimeLogsText.contains("resolved=MacBook Pro Microphone [built-in-mic]"))
    }

    @Test
    @MainActor
    func explicitInputDevicePreferencePassesSelectedUIDToCaptureService() {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let airPodsMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 202,
            uid: "airpods-mic",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        microphoneService.inputDevices = [builtInMicrophone, airPodsMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid

        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true
        model.settings.inputDevicePreference = .specificDeviceUID(airPodsMicrophone.uid)
        model.refreshAvailableInputDevices()

        model.startRecording()

        #expect(microphoneService.lastPreferredInputDeviceUID == airPodsMicrophone.uid)
        #expect(model.runtimeLogsText.contains("specific My AirPods Pro [airpods-mic]"))
    }

    @Test
    @MainActor
    func captureStartLogsSelectedInputDeviceAndFormat() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        microphoneService.inputDevices = [builtInMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid

        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true
        model.refreshAvailableInputDevices()

        model.startRecording()
        await settleVoiceCallbacks()

        #expect(model.runtimeLogsText.contains("[Mic] Starting capture preference=automatic (built-in preferred) resolved=MacBook Pro Microphone [built-in-mic]"))
        #expect(model.runtimeLogsText.contains("[Mic] Engine started device=MacBook Pro Microphone [built-in-mic]"))
        #expect(model.runtimeLogsText.contains("input=48000Hz/1ch/float32 output=16000Hz/1ch/int16"))
    }

    @Test
    @MainActor
    func audioDeviceChangeRefreshesAvailableInputDevices() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        microphoneService.inputDevices = [
            MicrophoneCaptureService.InputDevice(
                deviceID: 101,
                uid: "built-in-mic",
                name: "MacBook Pro Microphone",
                transport: .builtIn
            )
        ]

        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        #expect(model.availableInputDevices.count == 1)

        microphoneService.inputDevices.append(
            MicrophoneCaptureService.InputDevice(
                deviceID: 202,
                uid: "airpods-mic",
                name: "My AirPods Pro",
                transport: .bluetooth
            )
        )
        microphoneService.emitInputDevicesChanged()
        await settleVoiceCallbacks()

        #expect(model.availableInputDevices.count == 2)
        #expect(model.runtimeLogsText.contains("[Mic] Available input devices updated"))
    }

    @Test
    @MainActor
    func resetConversationClearsPendingComposerAttachment() throws {
        let model = AppModel()
        let imageURL = try makeTestImageURL()
        defer { try? FileManager.default.removeItem(at: imageURL.deletingLastPathComponent()) }

        model.pendingComposerImage = ComposerImageAttachment(
            id: UUID(),
            fileURL: imageURL,
            displayName: "sample.png"
        )

        model.resetConversation()

        #expect(model.pendingComposerImage == nil)
        #expect(model.canResetConversation == false)
    }

    @Test
    @MainActor
    func busyStateClearsAfterTurnFinished() {
        let model = AppModel()
        model.composerText = "First request"
        model.sendTextMessage()

        let turnID = model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last?.turnID

        #expect(model.isAwaitingAgentResponse)
        #expect(model.isComposerLocked)

        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: turnID,
            status: "finished",
            isFinal: true
        ))

        #expect(model.isAwaitingAgentResponse == false)
        #expect(model.isComposerLocked == false)
        #expect(model.statusText == "Ready")
    }

    @Test
    @MainActor
    func stopActiveTurnCancelsOnlyTheCurrentGeneratedTurn() {
        let model = AppModel()
        model.composerText = "First request"
        model.sendTextMessage()
        let turnID = model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last?.turnID
        model.composerText = "Queued follow-up"

        model.stopActiveTurn()

        #expect(model.isStopping)
        #expect(model.lastRuntimeCommand?.type == "cancel")
        #expect(model.lastRuntimeCommand?.turnID == turnID)
        #expect(model.isAwaitingAgentResponse)
        #expect(model.isComposerLocked)

        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: turnID,
            status: "cancelled",
            isFinal: true
        ))

        #expect(model.isStopping == false)
        #expect(model.isAwaitingAgentResponse == false)
        #expect(model.isComposerLocked == false)
        #expect(model.composerText == "Queued follow-up")
        #expect(model.statusText == "Stopped")
    }

    @Test
    @MainActor
    func spokenReplyKeepsOutputStoppableUntilPlaybackFinishes() throws {
        let runtime = FakeRuntimeClient()
        let playbackService = FakeAudioPlaybackService()
        let model = AppModel(runtime: runtime, playbackService: playbackService)
        model.composerText = "Speak the answer"
        model.sendTextMessage()

        let turnID = try #require(model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last?.turnID)

        model.handle(event: runtimeEvent(type: "assistant_delta", turnID: turnID, text: "Hello there"))
        model.handle(event: runtimeEvent(
            type: "audio_chunk",
            turnID: turnID,
            chunkBase64: Data([0x00, 0x01, 0x02, 0x03]).base64EncodedString(),
            sampleRate: 24_000,
            channels: 1,
            audioEncoding: "pcm_s16le"
        ))

        #expect(model.isReplySpeechActive)
        #expect(model.canStopCurrentOutput)
        #expect(model.statusText == "Speaking…")

        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: turnID,
            status: "finished",
            isFinal: true
        ))

        #expect(model.isAwaitingAgentResponse == false)
        #expect(model.canStopCurrentOutput)
        #expect(playbackService.finishStreamCallCount == 1)
        #expect(model.statusText == "Speaking…")

        playbackService.emitPlaybackFinished()

        #expect(model.isReplySpeechActive == false)
        #expect(model.canStopCurrentOutput == false)
        #expect(model.isComposerLocked == false)
        #expect(model.statusText == "Ready")
    }

    @Test
    @MainActor
    func stopDuringActiveReplyPlaybackCancelsTurnAndStopsAudioImmediately() throws {
        let runtime = FakeRuntimeClient()
        let playbackService = FakeAudioPlaybackService()
        let model = AppModel(runtime: runtime, playbackService: playbackService)
        model.composerText = "Speak the answer"
        model.sendTextMessage()

        let turnID = try #require(model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last?.turnID)

        model.handle(event: runtimeEvent(
            type: "audio_chunk",
            turnID: turnID,
            chunkBase64: Data([0x10, 0x11]).base64EncodedString(),
            sampleRate: 24_000,
            channels: 1,
            audioEncoding: "pcm_s16le"
        ))

        let commandCountBeforeStop = runtime.sentCommands.count
        let stopCountBeforeStop = playbackService.stopCallCount
        model.stopActiveTurn()

        #expect(playbackService.stopCallCount == stopCountBeforeStop + 1)
        #expect(runtime.sentCommands.count == commandCountBeforeStop + 1)
        #expect(runtime.lastSentCommand?.type == "cancel")
        #expect(runtime.lastSentCommand?.turnID == turnID)
        #expect(model.isStopping)
        #expect(model.canStopCurrentOutput)

        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: turnID,
            status: "cancelled",
            isFinal: true
        ))

        #expect(model.canStopCurrentOutput == false)
        #expect(model.statusText == "Stopped")
    }

    @Test
    @MainActor
    func stopAfterFinishedReplyPlaybackTailStopsLocallyWithoutCancellingAssistantText() throws {
        let runtime = FakeRuntimeClient()
        let playbackService = FakeAudioPlaybackService()
        let model = AppModel(runtime: runtime, playbackService: playbackService)
        model.composerText = "Speak the answer"
        model.sendTextMessage()

        let turnID = try #require(model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last?.turnID)

        model.handle(event: runtimeEvent(type: "assistant_delta", turnID: turnID, text: "Finished answer"))
        model.handle(event: runtimeEvent(
            type: "audio_chunk",
            turnID: turnID,
            chunkBase64: Data([0x20, 0x21]).base64EncodedString(),
            sampleRate: 24_000,
            channels: 1,
            audioEncoding: "pcm_s16le"
        ))
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: turnID,
            status: "finished",
            isFinal: true
        ))

        let commandCountBeforeStop = runtime.sentCommands.count
        let stopCountBeforeStop = playbackService.stopCallCount
        model.stopActiveTurn()

        #expect(playbackService.stopCallCount == stopCountBeforeStop + 1)
        #expect(runtime.sentCommands.count == commandCountBeforeStop)
        #expect(model.canStopCurrentOutput == false)
        #expect(model.statusText == "Ready")

        let finalAssistantMessage = try #require(assistantMessage(for: turnID, in: model))
        #expect(finalAssistantMessage.isCancelled == false)
        #expect(finalAssistantMessage.isStreaming == false)
    }

    @Test
    @MainActor
    func manualAssistantReadAloudKeepsExistingStopPath() throws {
        let runtime = FakeRuntimeClient()
        let playbackService = FakeAudioPlaybackService()
        let model = AppModel(runtime: runtime, playbackService: playbackService)
        let assistantTurnID = "assistant-turn"

        model.handle(event: runtimeEvent(type: "assistant_delta", turnID: assistantTurnID, text: "Read this aloud"))
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: assistantTurnID,
            status: "finished",
            isFinal: true
        ))

        let message = try #require(assistantMessage(for: assistantTurnID, in: model))
        model.speakAssistantMessage(message)
        let speechTurnID = try #require(runtime.lastSentCommand?.turnID)

        #expect(runtime.lastSentCommand?.type == "speak_text")
        #expect(model.canStopCurrentOutput == false)
        #expect(model.isSpeakingAssistantMessage(message))

        let stopCountBeforeStop = playbackService.stopCallCount
        model.stopActiveTurn()

        #expect(playbackService.stopCallCount == stopCountBeforeStop + 1)
        #expect(runtime.lastSentCommand?.type == "cancel")
        #expect(runtime.lastSentCommand?.turnID == speechTurnID)
        #expect(model.canStopCurrentOutput == false)
        #expect(model.isSpeakingAssistantMessage(message) == false)
    }

    @Test
    @MainActor
    func resetConversationAfterImageTurnKeepsNextTypedSendTextOnly() throws {
        let model = AppModel()
        let imageURL = try makeTestImageURL()
        defer { try? FileManager.default.removeItem(at: imageURL.deletingLastPathComponent()) }

        model.pendingComposerImage = ComposerImageAttachment(
            id: UUID(),
            fileURL: imageURL,
            displayName: "sample.png"
        )
        model.composerText = "What is in this image?"
        model.sendTextMessage()

        model.resetConversation()
        model.composerText = "Text only follow-up"
        model.sendTextMessage()

        let userMessages = model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }

        #expect(userMessages.count == 1)
        #expect(userMessages.last?.text == "Text only follow-up")
        #expect(userMessages.last?.imageAttachment == nil)
        #expect(model.lastRuntimeCommand?.type == "send_text")
        #expect(model.lastRuntimeCommand?.attachments == nil)
    }

    private func runtimeEvent(
        type: String,
        turnID: String? = nil,
        modelID: String? = nil,
        installableID: String? = nil,
        installState: String? = nil,
        warmState: String? = nil,
        message: String? = nil,
        stage: String? = nil,
        text: String? = nil,
        chunkBase64: String? = nil,
        sampleRate: Int? = nil,
        channels: Int? = nil,
        audioEncoding: String? = nil,
        bytesDownloaded: Int64? = nil,
        bytesTotal: Int64? = nil,
        etaSeconds: Double? = nil,
        speedBytesPerSecond: Double? = nil,
        toolCallID: String? = nil,
        toolName: String? = nil,
        toolSummary: String? = nil,
        toolArguments: [String: JSONValue]? = nil,
        approvalState: String? = nil,
        executionState: String? = nil,
        status: String? = nil,
        isFinal: Bool? = nil
    ) -> RuntimeEventEnvelope {
        RuntimeEventEnvelope(
            type: type,
            turnID: turnID,
            modelID: modelID,
            installableID: installableID,
            installState: installState,
            warmState: warmState,
            message: message,
            stage: stage,
            text: text,
            progress: nil,
            bytesDownloaded: bytesDownloaded,
            bytesTotal: bytesTotal,
            etaSeconds: etaSeconds,
            speedBytesPerSecond: speedBytesPerSecond,
            chunkBase64: chunkBase64,
            sampleRate: sampleRate,
            channels: channels,
            audioEncoding: audioEncoding,
            toolCallID: toolCallID,
            toolName: toolName,
            toolSummary: toolSummary,
            toolArguments: toolArguments,
            safetyClass: nil,
            approvalState: approvalState,
            executionState: executionState,
            output: nil,
            status: status,
            isFinal: isFinal
        )
    }

    private func makeTestImageURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let imageURL = directory.appendingPathComponent("sample.png")
        let size = NSSize(width: 12, height: 12)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.systemBlue.setFill()
        NSBezierPath(rect: NSRect(origin: .zero, size: size)).fill()
        image.unlockFocus()

        guard
            let tiff = image.tiffRepresentation,
            let bitmap = NSBitmapImageRep(data: tiff),
            let pngData = bitmap.representation(using: .png, properties: [:])
        else {
            throw CocoaError(.fileWriteUnknown)
        }

        try pngData.write(to: imageURL, options: .atomic)
        return imageURL
    }

    @MainActor
    private func currentVoiceDraft(in model: AppModel) -> UserVoiceDraft? {
        model.conversation.compactMap { item -> UserVoiceDraft? in
            guard case .userVoiceDraft(let draft) = item else { return nil }
            return draft
        }.last
    }

    @MainActor
    private func userMessage(for turnID: String, in model: AppModel) -> UserMessage? {
        model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item, message.turnID == turnID else { return nil }
            return message
        }.last
    }

    @MainActor
    private func assistantMessage(for turnID: String, in model: AppModel) -> AssistantMessage? {
        model.conversation.compactMap { item -> AssistantMessage? in
            guard case .assistantMessage(let message) = item, message.turnID == turnID else { return nil }
            return message
        }.last
    }

    @MainActor
    private func systemStatuses(in model: AppModel) -> [SystemStatusMessage] {
        model.conversation.compactMap { item -> SystemStatusMessage? in
            guard case .systemStatus(let status) = item else { return nil }
            return status
        }
    }

    @MainActor
    private func settleVoiceCallbacks() async {
        for _ in 0..<2 {
            await Task.yield()
        }
    }
}

private final class FakeMicrophoneCaptureService: MicrophoneCaptureServicing {
    var onInputDevicesChanged: (@Sendable () -> Void)?
    var status: MicrophoneCaptureService.AuthorizationStatus = .authorized
    var requestAccessResult = true
    var inputDevices: [MicrophoneCaptureService.InputDevice] = []
    var defaultInputDeviceUID: String?
    var startResults: [Result<Void, Error>] = []
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0
    private(set) var lastPreferredInputDeviceUID: String?
    private(set) var lastStartedInputDevice: MicrophoneCaptureService.InputDevice?

    private var onStarted: (@Sendable () -> Void)?
    private var onFirstInputBuffer: (@Sendable () -> Void)?
    private var onFirstChunk: (@Sendable () -> Void)?
    private var onError: (@Sendable (Error) -> Void)?
    private var logHandler: (@Sendable (String) -> Void)?
    private var chunkHandler: (@Sendable (MicrophoneCaptureService.CaptureChunk) -> Void)?
    private var hasDeliveredFirstInputBuffer = false
    private var hasDeliveredFirstChunk = false

    func authorizationStatus() -> MicrophoneCaptureService.AuthorizationStatus {
        status
    }

    func requestAccess() async -> Bool {
        requestAccessResult
    }

    func availableInputDevices() -> [MicrophoneCaptureService.InputDevice] {
        inputDevices
    }

    func defaultInputDevice() -> MicrophoneCaptureService.InputDevice? {
        guard let defaultInputDeviceUID else {
            return inputDevices.first
        }
        return inputDevices.first(where: { $0.uid == defaultInputDeviceUID }) ?? inputDevices.first
    }

    func start(
        preferredInputDeviceUID: String?,
        onStarted: (@Sendable () -> Void)?,
        onFirstInputBuffer: (@Sendable () -> Void)?,
        onFirstChunk: (@Sendable () -> Void)?,
        onError: (@Sendable (Error) -> Void)?,
        logHandler: (@Sendable (String) -> Void)?,
        chunkHandler: @escaping @Sendable (MicrophoneCaptureService.CaptureChunk) -> Void
    ) throws {
        startCallCount += 1
        lastPreferredInputDeviceUID = preferredInputDeviceUID
        lastStartedInputDevice = {
            if let preferredInputDeviceUID {
                return inputDevices.first(where: { $0.uid == preferredInputDeviceUID })
            }
            return defaultInputDevice()
        }()
        hasDeliveredFirstInputBuffer = false
        hasDeliveredFirstChunk = false
        if !startResults.isEmpty {
            let result = startResults.removeFirst()
            if case .failure(let error) = result {
                logHandler?(
                    "[Mic] Failed to start capture requested=\(describe(device: inputDevices.first(where: { $0.uid == preferredInputDeviceUID }))) " +
                    "resolved=\(describe(device: lastStartedInputDevice)) " +
                    "input=48000Hz/1ch/float32: \(error.localizedDescription)"
                )
                throw error
            }
        }
        self.onStarted = onStarted
        self.onFirstInputBuffer = onFirstInputBuffer
        self.onFirstChunk = onFirstChunk
        self.onError = onError
        self.logHandler = logHandler
        self.chunkHandler = chunkHandler
        logHandler?(
            "[Mic] Engine started device=\(describe(device: lastStartedInputDevice)) " +
            "requested=\(describe(device: inputDevices.first(where: { $0.uid == preferredInputDeviceUID }))) " +
            "input=48000Hz/1ch/float32 output=16000Hz/1ch/int16"
        )
        onStarted?()
    }

    func stop() {
        stopCallCount += 1
        onStarted = nil
        onFirstInputBuffer = nil
        onFirstChunk = nil
        onError = nil
        logHandler?("[Mic] Stopped capture after 0ms (rawInput=\(hasDeliveredFirstInputBuffer), convertedChunk=\(hasDeliveredFirstChunk))")
        logHandler = nil
        chunkHandler = nil
        hasDeliveredFirstInputBuffer = false
        hasDeliveredFirstChunk = false
    }

    func emitInputDevicesChanged() {
        onInputDevicesChanged?()
    }

    func emitInputBuffer() {
        guard !hasDeliveredFirstInputBuffer else { return }
        hasDeliveredFirstInputBuffer = true
        logHandler?("[Mic] First input buffer after 30ms")
        onFirstInputBuffer?()
    }

    func emitChunk(_ chunk: MicrophoneCaptureService.CaptureChunk = .init(data: Data([0x00, 0x01, 0x02, 0x03]), sampleRate: 16_000, channels: 1)) {
        if !hasDeliveredFirstInputBuffer {
            emitInputBuffer()
        }
        if !hasDeliveredFirstChunk {
            hasDeliveredFirstChunk = true
            logHandler?("[Mic] First converted chunk after 45ms output=16000Hz/1ch/int16")
            onFirstChunk?()
        }
        chunkHandler?(chunk)
    }

    func emitError(_ error: Error) {
        logHandler?("[Mic] Capture error after 0ms: \(error.localizedDescription)")
        onError?(error)
    }

    private func describe(device: MicrophoneCaptureService.InputDevice?) -> String {
        guard let device else { return "system-default" }
        return "\(device.name) [\(device.uid)]"
    }
}

@MainActor
private final class FakeAudioPlaybackService: AudioPlaybackServicing {
    var onPlaybackStarted: (() -> Void)?
    var onPlaybackFinished: (() -> Void)?
    private(set) var enqueuedChunkCount = 0
    private(set) var finishStreamCallCount = 0
    private(set) var stopCallCount = 0

    func enqueuePCMData(_ data: Data, sampleRate: Int, channels: Int, encoding: String) {
        enqueuedChunkCount += 1
    }

    func finishStream() {
        finishStreamCallCount += 1
    }

    func stop() {
        stopCallCount += 1
    }

    func emitPlaybackStarted() {
        onPlaybackStarted?()
    }

    func emitPlaybackFinished() {
        onPlaybackFinished?()
    }
}

@MainActor
private final class FakeRuntimeClient: RuntimeClienting {
    var onEvent: (@Sendable @MainActor (RuntimeEventEnvelope) -> Void)?
    var onError: (@Sendable @MainActor (String) -> Void)?
    var onLog: (@Sendable @MainActor (String) -> Void)?
    var onTermination: (@Sendable @MainActor (String) -> Void)?
    var lastSentCommand: RuntimeCommandEnvelope?
    var sentCommands: [RuntimeCommandEnvelope] = []
    var isRunning = true

    func start() throws {}

    func stop(gracefully: Bool) {}

    func send(_ command: RuntimeCommandEnvelope) throws {
        lastSentCommand = command
        sentCommands.append(command)
    }
}
