import AudioToolbox
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
    @MainActor
    func assistantSegmentIDsCreateSeparateBubblesInterleavedWithTools() {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: "turn-1",
            text: "Checking the current state…",
            assistantSegmentID: "segment-1",
            isFinalSegment: false
        ))
        model.handle(event: runtimeEvent(
            type: "tool_proposed",
            turnID: "turn-1",
            toolCallID: "tool-1",
            toolName: "execute_script",
            toolSummary: "Inspect Finder",
            toolArguments: ["script_content": .string("return system info")],
            safetyClass: SafetyClass.readOnly.rawValue,
            approvalState: ApprovalState.notRequired.rawValue,
            executionState: ExecutionState.proposed.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: "turn-1",
            text: "Here is the result.",
            assistantSegmentID: "segment-2",
            isFinalSegment: true
        ))

        let itemKinds = model.conversation.map { item -> String in
            switch item {
            case .assistantMessage:
                return "assistant"
            case .toolInvocation:
                return "tool"
            default:
                return "other"
            }
        }
        let assistantMessages = model.conversation.compactMap { item -> AssistantMessage? in
            guard case .assistantMessage(let message) = item else { return nil }
            return message
        }

        #expect(itemKinds == ["assistant", "tool", "assistant"])
        #expect(assistantMessages.map(\.segmentID) == ["segment-1", "segment-2"])
        #expect(assistantMessages.map(\.isFinalSegment) == [false, true])
    }

    @Test
    @MainActor
    func assistantDeltaFlushesBufferedSegmentTextBeforeToolCards() {
        let model = AppModel(assistantDeltaFlushDelay: .seconds(60))

        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: "turn-1",
            text: "Searching",
            assistantSegmentID: "segment-1",
            isFinalSegment: false
        ))
        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: "turn-1",
            text: " Safari",
            assistantSegmentID: "segment-1",
            isFinalSegment: false
        ))

        let assistantBeforeFlush = model.conversation.compactMap { item -> AssistantMessage? in
            guard case .assistantMessage(let message) = item else { return nil }
            return message
        }
        #expect(assistantBeforeFlush.count == 1)
        #expect(assistantBeforeFlush.first?.text == "Searching")

        model.handle(event: runtimeEvent(
            type: "tool_proposed",
            turnID: "turn-1",
            toolCallID: "tool-1",
            toolName: "execute_script",
            toolSummary: "Drive Safari",
            toolArguments: ["script_content": .string("tell application \"Safari\" to activate")],
            safetyClass: SafetyClass.readOnly.rawValue,
            approvalState: ApprovalState.notRequired.rawValue,
            executionState: ExecutionState.proposed.rawValue
        ))

        let itemKinds = model.conversation.map { item -> String in
            switch item {
            case .assistantMessage:
                return "assistant"
            case .toolInvocation:
                return "tool"
            default:
                return "other"
            }
        }
        let assistantAfterFlush = model.conversation.compactMap { item -> AssistantMessage? in
            guard case .assistantMessage(let message) = item else { return nil }
            return message
        }

        #expect(itemKinds == ["assistant", "tool"])
        #expect(assistantAfterFlush.first?.text == "Searching Safari")
        #expect(assistantAfterFlush.first?.isStreaming == true)
    }

    @Test
    @MainActor
    func repeatedToolUpdatesMutateExistingToolCardInsteadOfAppendingMoreCards() throws {
        let model = AppModel()

        model.handle(event: runtimeEvent(
            type: "tool_proposed",
            turnID: "turn-1",
            toolCallID: "tool-1",
            toolName: "execute_script",
            toolSummary: "Inspect Safari",
            toolArguments: ["script_content": .string("return 1")],
            safetyClass: SafetyClass.readOnly.rawValue,
            approvalState: ApprovalState.notRequired.rawValue,
            executionState: ExecutionState.proposed.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "tool_output",
            turnID: "turn-1",
            toolCallID: "tool-1",
            output: "partial output"
        ))
        model.handle(event: runtimeEvent(
            type: "tool_finished",
            turnID: "turn-1",
            toolCallID: "tool-1",
            executionState: ExecutionState.finished.rawValue,
            output: "final output"
        ))

        let toolMessages = model.conversation.compactMap { item -> ToolInvocation? in
            guard case .toolInvocation(let tool) = item else { return nil }
            return tool
        }
        let tool = try #require(toolMessages.first)

        #expect(toolMessages.count == 1)
        #expect(tool.output == "final output")
        #expect(tool.executionState == .finished)
    }

    @Test
    @MainActor
    func rerunHistoryIncludesOnlyFinalAssistantSegment() throws {
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime)

        model.composerText = "First turn"
        model.sendTextMessage()
        let firstTurnID = try #require(model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last?.turnID)
        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: firstTurnID,
            text: "Thinking aloud",
            assistantSegmentID: "segment-intermediate",
            isFinalSegment: false
        ))
        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: firstTurnID,
            text: "Final answer",
            assistantSegmentID: "segment-final",
            isFinalSegment: true
        ))
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: firstTurnID,
            status: "finished",
            isFinal: true
        ))

        model.composerText = "Second turn"
        model.sendTextMessage()
        let secondMessage = try #require(model.conversation.compactMap { item -> UserMessage? in
            guard case .userMessage(let message) = item else { return nil }
            return message
        }.last)
        model.handle(event: runtimeEvent(
            type: "turn_finished",
            turnID: secondMessage.turnID,
            status: "finished",
            isFinal: true
        ))

        model.editUserMessage(secondMessage, newText: "Second turn updated")

        let replaceHistoryCommand = try #require(runtime.sentCommands.first(where: { $0.type == "replace_history" }))
        #expect(replaceHistoryCommand.history?.count == 2)
        #expect(replaceHistoryCommand.history?[0].text == "First turn")
        #expect(replaceHistoryCommand.history?[1].text == "Final answer")
    }

    @Test
    @MainActor
    func replySpeechStartsOnlyAfterFinalAssistantSegment() throws {
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
            type: "assistant_delta",
            turnID: turnID,
            text: "Checking something first…",
            assistantSegmentID: "segment-1",
            isFinalSegment: false
        ))
        model.handle(event: runtimeEvent(
            type: "audio_chunk",
            turnID: turnID,
            chunkBase64: Data([0x00, 0x01]).base64EncodedString(),
            sampleRate: 24_000,
            channels: 1,
            audioEncoding: "pcm_s16le"
        ))

        #expect(model.isReplySpeechActive == false)

        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: turnID,
            text: "Done.",
            assistantSegmentID: "segment-2",
            isFinalSegment: true
        ))
        model.handle(event: runtimeEvent(
            type: "audio_chunk",
            turnID: turnID,
            chunkBase64: Data([0x02, 0x03]).base64EncodedString(),
            sampleRate: 24_000,
            channels: 1,
            audioEncoding: "pcm_s16le"
        ))

        #expect(model.isReplySpeechActive)
        #expect(model.statusText == "Speaking…")
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
    func serviceStartsCaptureUsingRequestedInputDeviceUID() throws {
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
        let controllerFactory = FakeCaptureControllerFactory()
        let controller = FakeCaptureController(
            boundInputDeviceID: airPodsMicrophone.deviceID,
            boundInputDeviceUID: airPodsMicrophone.uid,
            boundInputDeviceName: airPodsMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)
        )
        controllerFactory.queuedControllers = [controller]
        let logs = LockedLogMessages()
        let started = LockedCounter()
        let service = MicrophoneCaptureService(
            authorizationStatusProvider: { .authorized },
            requestAccessHandler: { true },
            availableInputDevicesProvider: { [builtInMicrophone, airPodsMicrophone] },
            defaultInputDeviceProvider: { builtInMicrophone },
            captureControllerFactory: controllerFactory
        )

        try service.start(
            preferredInputDeviceUID: airPodsMicrophone.uid,
            onStarted: { started.increment() },
            onFirstInputBuffer: nil,
            onFirstChunk: nil,
            onError: nil,
            logHandler: { logs.append($0) },
            chunkHandler: { _ in }
        )

        #expect(controllerFactory.selectedInputDeviceUIDs == [airPodsMicrophone.uid])
        #expect(started.value == 1)
        #expect(logs.contains("Capture session started device=My AirPods Pro [airpods-mic]"))
        #expect(logs.contains("selected=My AirPods Pro [airpods-mic]"))
        service.stop()
        #expect(controller.stopCallCount == 1)
    }

    @Test
    func serviceFallsBackCleanlyWhenPreferredUIDIsUnavailable() throws {
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let controllerFactory = FakeCaptureControllerFactory()
        let controller = FakeCaptureController(
            boundInputDeviceID: builtInMicrophone.deviceID,
            boundInputDeviceUID: builtInMicrophone.uid,
            boundInputDeviceName: builtInMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)
        )
        controllerFactory.queuedControllers = [controller]
        let logs = LockedLogMessages()
        let service = MicrophoneCaptureService(
            authorizationStatusProvider: { .authorized },
            requestAccessHandler: { true },
            availableInputDevicesProvider: { [builtInMicrophone] },
            defaultInputDeviceProvider: { builtInMicrophone },
            captureControllerFactory: controllerFactory
        )

        try service.start(
            preferredInputDeviceUID: "missing-mic",
            onStarted: nil,
            onFirstInputBuffer: nil,
            onFirstChunk: nil,
            onError: nil,
            logHandler: { logs.append($0) },
            chunkHandler: { _ in }
        )

        #expect(controllerFactory.selectedInputDeviceUIDs == [builtInMicrophone.uid])
        #expect(logs.contains("Requested input device uid=missing-mic is unavailable. Falling back to the active system route."))
        #expect(logs.contains("Capture session started device=MacBook Pro Microphone [built-in-mic]"))
        #expect(logs.contains("selected=MacBook Pro Microphone [built-in-mic]"))
        service.stop()
    }

    @Test
    func serviceEmitsStartedAndFirstBufferCallbacksExactlyOnce() throws {
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let controllerFactory = FakeCaptureControllerFactory()
        let controller = FakeCaptureController(
            boundInputDeviceID: builtInMicrophone.deviceID,
            boundInputDeviceUID: builtInMicrophone.uid,
            boundInputDeviceName: builtInMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)
        )
        controllerFactory.queuedControllers = [controller]
        let started = LockedCounter()
        let firstInputBuffer = LockedCounter()
        let firstChunk = LockedCounter()
        let chunkRecorder = LockedChunkRecorder()
        let service = MicrophoneCaptureService(
            authorizationStatusProvider: { .authorized },
            requestAccessHandler: { true },
            availableInputDevicesProvider: { [builtInMicrophone] },
            defaultInputDeviceProvider: { builtInMicrophone },
            captureControllerFactory: controllerFactory
        )

        try service.start(
            preferredInputDeviceUID: builtInMicrophone.uid,
            onStarted: { started.increment() },
            onFirstInputBuffer: { firstInputBuffer.increment() },
            onFirstChunk: { firstChunk.increment() },
            onError: nil,
            logHandler: nil,
            chunkHandler: { chunkRecorder.append($0) }
        )

        controller.emit(buffer: makeFloatPCMBuffer(sampleRate: 48_000, frameCount: 4_800))
        controller.emit(buffer: makeFloatPCMBuffer(sampleRate: 48_000, frameCount: 4_800))

        #expect(started.value == 1)
        #expect(firstInputBuffer.value == 1)
        #expect(firstChunk.value == 1)
        #expect(chunkRecorder.chunks.count == 2)
        service.stop()
    }

    @Test
    func serviceConverts24kAnd48kMonoInputTo16kInt16Chunks() throws {
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let controllerFactory = FakeCaptureControllerFactory()
        let firstController = FakeCaptureController(
            boundInputDeviceID: builtInMicrophone.deviceID,
            boundInputDeviceUID: builtInMicrophone.uid,
            boundInputDeviceName: builtInMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)
        )
        let secondController = FakeCaptureController(
            boundInputDeviceID: builtInMicrophone.deviceID,
            boundInputDeviceUID: builtInMicrophone.uid,
            boundInputDeviceName: builtInMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)
        )
        controllerFactory.queuedControllers = [firstController, secondController]
        let chunkRecorder = LockedChunkRecorder()
        let service = MicrophoneCaptureService(
            authorizationStatusProvider: { .authorized },
            requestAccessHandler: { true },
            availableInputDevicesProvider: { [builtInMicrophone] },
            defaultInputDeviceProvider: { builtInMicrophone },
            captureControllerFactory: controllerFactory
        )

        try service.start(
            preferredInputDeviceUID: builtInMicrophone.uid,
            onStarted: nil,
            onFirstInputBuffer: nil,
            onFirstChunk: nil,
            onError: nil,
            logHandler: nil,
            chunkHandler: { chunkRecorder.append($0) }
        )
        firstController.emit(buffer: makeFloatPCMBuffer(sampleRate: 24_000, frameCount: 2_400))
        service.stop()

        try service.start(
            preferredInputDeviceUID: builtInMicrophone.uid,
            onStarted: nil,
            onFirstInputBuffer: nil,
            onFirstChunk: nil,
            onError: nil,
            logHandler: nil,
            chunkHandler: { chunkRecorder.append($0) }
        )
        secondController.emit(buffer: makeFloatPCMBuffer(sampleRate: 48_000, frameCount: 4_800))
        service.stop()

        let chunks = chunkRecorder.chunks
        #expect(chunks.count == 2)
        #expect(chunks.allSatisfy { $0.sampleRate == 16_000 && $0.channels == 1 })
        let firstDuration = Double(chunks[0].data.count / MemoryLayout<Int16>.size) / 16_000
        let secondDuration = Double(chunks[1].data.count / MemoryLayout<Int16>.size) / 16_000
        #expect(abs(firstDuration - 0.1) < 0.02)
        #expect(abs(secondDuration - 0.1) < 0.02)
    }

    @Test
    func serviceForwardsRuntimeErrorsAndCanRestartAfterStop() throws {
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let controllerFactory = FakeCaptureControllerFactory()
        let firstController = FakeCaptureController(
            boundInputDeviceID: builtInMicrophone.deviceID,
            boundInputDeviceUID: builtInMicrophone.uid,
            boundInputDeviceName: builtInMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)
        )
        let secondController = FakeCaptureController(
            boundInputDeviceID: builtInMicrophone.deviceID,
            boundInputDeviceUID: builtInMicrophone.uid,
            boundInputDeviceName: builtInMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)
        )
        controllerFactory.queuedControllers = [firstController, secondController]
        let errors = LockedLogMessages()
        let service = MicrophoneCaptureService(
            authorizationStatusProvider: { .authorized },
            requestAccessHandler: { true },
            availableInputDevicesProvider: { [builtInMicrophone] },
            defaultInputDeviceProvider: { builtInMicrophone },
            captureControllerFactory: controllerFactory
        )

        try service.start(
            preferredInputDeviceUID: builtInMicrophone.uid,
            onStarted: nil,
            onFirstInputBuffer: nil,
            onFirstChunk: nil,
            onError: { errors.append($0.localizedDescription) },
            logHandler: nil,
            chunkHandler: { _ in }
        )

        firstController.emitError(NSError(
            domain: "MacAssistantTests.Microphone",
            code: 99,
            userInfo: [NSLocalizedDescriptionKey: "Session interrupted."]
        ))
        #expect(errors.contains("Session interrupted."))

        service.stop()
        #expect(firstController.stopCallCount == 1)

        try service.start(
            preferredInputDeviceUID: builtInMicrophone.uid,
            onStarted: nil,
            onFirstInputBuffer: nil,
            onFirstChunk: nil,
            onError: nil,
            logHandler: nil,
            chunkHandler: { _ in }
        )

        #expect(secondController.startCallCount == 1)
        service.stop()
        #expect(secondController.stopCallCount == 1)
    }

    @Test
    func serviceLogsBackendBannerAndRouteSummary() throws {
        MicrophoneCaptureService.resetBackendBannerForTesting()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let airPodsOutput = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 301,
            uid: "airpods-output",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        let controllerFactory = FakeCaptureControllerFactory()
        let controller = FakeCaptureController(
            boundInputDeviceID: builtInMicrophone.deviceID,
            boundInputDeviceUID: builtInMicrophone.uid,
            boundInputDeviceName: builtInMicrophone.name,
            deviceInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1),
            clientInputFormat: AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)
        )
        controllerFactory.queuedControllers = [controller]
        let logs = LockedLogMessages()
        let service = MicrophoneCaptureService(
            authorizationStatusProvider: { .authorized },
            requestAccessHandler: { true },
            availableInputDevicesProvider: { [builtInMicrophone] },
            defaultInputDeviceProvider: { builtInMicrophone },
            routeStateProvider: {
                .init(defaultOutputDevice: airPodsOutput, isArbitrationActive: true)
            },
            buildInfoProvider: {
                (version: "test-version", build: "test-build")
            },
            captureControllerFactory: controllerFactory
        )

        try service.start(
            preferredInputDeviceUID: builtInMicrophone.uid,
            onStarted: nil,
            onFirstInputBuffer: nil,
            onFirstChunk: nil,
            onError: nil,
            logHandler: { logs.append($0) },
            chunkHandler: { _ in }
        )

        #expect(logs.contains("[Mic] backend=auhal version=test-version build=test-build"))
        #expect(logs.contains("Capture session started device=MacBook Pro Microphone [built-in-mic]"))
        #expect(logs.contains("selected=MacBook Pro Microphone [built-in-mic]"))
        #expect(logs.contains("defaultInput=MacBook Pro Microphone [built-in-mic]"))
        #expect(logs.contains("defaultOutput=My AirPods Pro [airpods-output]"))
        #expect(logs.contains("splitRouteArbitration=engaged"))
        #expect(logs.contains("deviceFormat=44100Hz/1ch/float32"))
        #expect(logs.contains("clientFormat=44100Hz/1ch/float32"))
        service.stop()
    }

    @Test
    func auhalControllerBindsRequestedDeviceAndRendersInputCallback() async throws {
        let airPodsMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 202,
            uid: "airpods-mic",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        let fakeUnit = FakeAUHALInputUnit(
            currentDeviceIDValue: airPodsMicrophone.deviceID,
            inputDeviceFormatValue: makeFloatStreamDescription(sampleRate: 24_000, channels: 1)
        )
        let recorder = LockedPCMBufferRecorder()
        let controller = try AUHALMicrophoneCaptureControllerFactory(
            inputUnitFactory: { fakeUnit }
        ).makeController(
            selectedInputDevice: airPodsMicrophone,
            onPCMBuffer: { recorder.append($0) },
            onError: { error in
                Issue.record("Unexpected AUHAL callback error: \(error.localizedDescription)")
            }
        )

        #expect(fakeUnit.setInputEnabledValues == [true])
        #expect(fakeUnit.setOutputEnabledValues == [false])
        #expect(fakeUnit.setCurrentDeviceValues == [airPodsMicrophone.deviceID])
        #expect(controller.boundInputDeviceID == airPodsMicrophone.deviceID)
        #expect(controller.boundInputDeviceUID == airPodsMicrophone.uid)
        #expect(controller.boundInputDeviceName == airPodsMicrophone.name)
        #expect(abs((controller.deviceInputFormat?.sampleRate ?? 0) - 24_000) < 0.5)
        #expect(abs((controller.clientInputFormat?.sampleRate ?? 0) - 24_000) < 0.5)

        try controller.start()
        #expect(fakeUnit.initializeCallCount == 1)
        #expect(fakeUnit.startCallCount == 1)

        fakeUnit.invokeInputCallback(frameCount: 2_400)
        try? await Task.sleep(for: .milliseconds(10))

        let buffers = recorder.buffers
        #expect(buffers.count == 1)
        #expect(abs(buffers[0].format.sampleRate - 24_000) < 0.5)
        #expect(buffers[0].frameLength == 2_400)

        controller.stop()
        #expect(fakeUnit.stopCallCount == 1)
        #expect(fakeUnit.uninitializeCallCount == 1)
    }

    @Test
    func auhalControllerPropagatesStartErrors() throws {
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let fakeUnit = FakeAUHALInputUnit(
            currentDeviceIDValue: builtInMicrophone.deviceID,
            inputDeviceFormatValue: makeFloatStreamDescription(sampleRate: 44_100, channels: 1)
        )
        fakeUnit.startError = NSError(
            domain: NSOSStatusErrorDomain,
            code: -10868,
            userInfo: [NSLocalizedDescriptionKey: "The operation couldn’t be completed. (OSStatus error -10868.)"]
        )

        let controller = try AUHALMicrophoneCaptureControllerFactory(
            inputUnitFactory: { fakeUnit }
        ).makeController(
            selectedInputDevice: builtInMicrophone,
            onPCMBuffer: { _ in },
            onError: { _ in }
        )

        do {
            try controller.start()
            Issue.record("Expected controller.start() to throw.")
        } catch {
            let nsError = error as NSError
            #expect(nsError.domain == NSOSStatusErrorDomain)
            #expect(nsError.code == -10868)
        }
        #expect(fakeUnit.initializeCallCount == 1)
        #expect(fakeUnit.uninitializeCallCount == 1)
        #expect(fakeUnit.startCallCount == 1)
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
        firstModel.settings.streamReplySpeechWhileGenerating = false

        let secondModel = AppModel(appSupportURL: tempDirectory, persistSettings: true)

        #expect(secondModel.settings.defaultVoicePreset == "neutral_female")
        #expect(secondModel.settings.inputDevicePreference == .specificDeviceUID("built-in-mic"))
        #expect(secondModel.settings.launchOnOpen)
        #expect(secondModel.settings.alwaysAcceptToolCalls)
        #expect(secondModel.settings.streamReplySpeechWhileGenerating == false)
    }

    @Test
    @MainActor
    func settingsDefaultToStreamingSpokenReplies() {
        let model = AppModel()

        #expect(model.settings.streamReplySpeechWhileGenerating)
    }

    @Test
    @MainActor
    func settingsGeneralSnapshotReflectsCurrentSelectionsAndDevices() {
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 1,
            uid: "built-in-mic",
            name: "MacBook Microphone",
            transport: .builtIn
        )
        let airPodsMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 2,
            uid: "airpods-mic",
            name: "AirPods Pro",
            transport: .bluetooth
        )
        let microphoneService = FakeMicrophoneCaptureService()
        microphoneService.inputDevices = [builtInMicrophone, airPodsMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid

        let model = AppModel(
            runtime: FakeRuntimeClient(),
            microphoneService: microphoneService
        )
        model.settings.defaultVoicePreset = "neutral_female"
        model.settings.inputDevicePreference = .specificDeviceUID("missing-mic")
        model.settings.launchOnOpen = true
        model.settings.alwaysAcceptToolCalls = true
        model.settings.streamReplySpeechWhileGenerating = false

        let snapshot = model.settingsGeneralSnapshot

        #expect(snapshot.defaultVoicePreset == "neutral_female")
        #expect(snapshot.selectedInputDevicePickerValue == "missing-mic")
        #expect(snapshot.launchOnOpen)
        #expect(snapshot.alwaysAcceptToolCalls)
        #expect(snapshot.streamReplySpeechWhileGenerating == false)
        #expect(snapshot.availableInputDevices.map(\.uid) == [builtInMicrophone.uid, airPodsMicrophone.uid])
        #expect(snapshot.unavailableSelectedInputDeviceUID == "missing-mic")
    }

    @Test
    @MainActor
    func settingsModelFilesSnapshotPreservesStableIDsAndMapsState() {
        let model = AppModel(runtime: FakeRuntimeClient())

        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.agentModel.rawValue,
            installState: ModelInstallState.installed.rawValue,
            warmState: WarmState.warm.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.ttsModel.rawValue,
            installState: ModelInstallState.downloading.rawValue,
            warmState: WarmState.cold.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "model_state",
            modelID: RuntimeModelID.sttModel.rawValue,
            installState: ModelInstallState.downloading.rawValue,
            warmState: WarmState.cold.rawValue
        ))
        model.handle(event: runtimeEvent(
            type: "download_progress",
            modelID: RuntimeModelID.ttsModel.rawValue,
            installableID: "voice_pack",
            bytesDownloaded: 512,
            bytesTotal: 1024,
            etaSeconds: 4.0,
            speedBytesPerSecond: 128
        ))

        let snapshot = model.settingsModelFilesSnapshot
        let agentEntry = snapshot.entries.first(where: { $0.id == RuntimeModelID.agentModel.rawValue })
        let ttsEntry = snapshot.entries.first(where: { $0.id == RuntimeModelID.ttsModel.rawValue })

        #expect(snapshot.entries.map(\.id) == RuntimeModelID.allCases.map(\.rawValue))
        #expect(agentEntry?.action == .deleteModel(.agentModel))
        #expect(agentEntry?.actionTitle == "Delete")
        #expect(agentEntry?.warmState == .warm)
        #expect(ttsEntry?.installState == .downloading)
        #expect(ttsEntry?.action == nil)
        #expect(ttsEntry?.progress?.bytesDownloaded == 512)
    }

    @Test
    @MainActor
    func settingsModelFileActionDispatchesInstallRequests() {
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime)

        model.handleSettingsModelFileAction(.installInstallable("agent_model"))

        #expect(runtime.sentCommands.last?.type == "download_model")
        #expect(runtime.sentCommands.last?.installableID == "agent_model")
        #expect(model.underlyingModels[RuntimeModelID.agentModel.rawValue]?.installState == .downloading)
    }

    @Test
    @MainActor
    func settingsRuntimeLogsSnapshotMatchesRetainedBuffer() {
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime)

        for index in 0..<12 {
            runtime.onLog?("settings-log-\(index)")
        }

        let snapshot = model.settingsRuntimeLogsSnapshot

        #expect(snapshot.entries == model.runtimeLogEntries)
        #expect(snapshot.lineCount == model.runtimeLogLineCount)
        #expect(snapshot.byteCount == model.runtimeLogByteCount)
        #expect(snapshot.entries.last?.text == "settings-log-11")
    }

    @Test
    @MainActor
    func runtimeLogsRetainNewestEntriesWithinLineCap() {
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime)

        for index in 0..<1_200 {
            runtime.onLog?("log-\(index)-" + String(repeating: "x", count: 32))
        }

        #expect(model.runtimeLogLineCount == 1_000)
        #expect(model.runtimeLogEntries.count == 1_000)
        #expect(model.runtimeLogEntries.first?.text.contains("log-200-") == true)
        #expect(model.runtimeLogEntries.last?.text.contains("log-1199-") == true)
        #expect(model.runtimeLogsText.contains("log-1199-"))
        #expect(!model.runtimeLogsText.contains("log-199-"))
        #expect(model.runtimeLogsText == model.runtimeLogEntries.map(\.text).joined(separator: "\n"))
    }

    @Test
    @MainActor
    func runtimeLogsTrimToByteLimitAndPreserveNewestEntries() {
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime)
        let oversizedPayload = String(repeating: "z", count: 12_000)

        for index in 0..<140 {
            runtime.onLog?("byte-log-\(index)-\(oversizedPayload)")
        }

        #expect(model.runtimeLogByteCount <= 1_048_576)
        #expect(model.runtimeLogEntries.last?.text.contains("byte-log-139-") == true)
        #expect(model.runtimeLogsText.contains("byte-log-139-"))
    }

    @Test
    @MainActor
    func startPrunesOnlyStaleLegacyAudioCacheFiles() throws {
        let runtime = FakeRuntimeClient()
        let appSupportURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let cacheURL = appSupportURL.appendingPathComponent("cache", isDirectory: true)

        try FileManager.default.createDirectory(at: cacheURL, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: appSupportURL)
        }

        let staleWAV = cacheURL.appendingPathComponent("stale.wav")
        let recentWAV = cacheURL.appendingPathComponent("recent.wav")
        let untouchedText = cacheURL.appendingPathComponent("note.txt")

        try Data("stale".utf8).write(to: staleWAV)
        try Data("recent".utf8).write(to: recentWAV)
        try Data("note".utf8).write(to: untouchedText)

        let staleDate = Date(timeIntervalSinceNow: -(8 * 24 * 60 * 60))
        let recentDate = Date()
        try FileManager.default.setAttributes([.modificationDate: staleDate], ofItemAtPath: staleWAV.path)
        try FileManager.default.setAttributes([.modificationDate: recentDate], ofItemAtPath: recentWAV.path)

        let model = AppModel(
            appSupportURL: appSupportURL,
            persistSettings: false,
            runtime: runtime
        )
        model.start()

        #expect(!FileManager.default.fileExists(atPath: staleWAV.path))
        #expect(FileManager.default.fileExists(atPath: recentWAV.path))
        #expect(FileManager.default.fileExists(atPath: untouchedText.path))
        #expect(model.runtimeLogsText.contains("[Maintenance] Removed 1 stale cached audio file"))
    }

    @Test
    func runtimeLaunchLogTrimmerRetainsNewestCompleteLinesWithinLimit() throws {
        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let logURL = tempDirectory.appendingPathComponent("runtime-launch.log")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: tempDirectory)
        }

        let content = (0..<40)
            .map { "entry-\($0)-" + String(repeating: "q", count: 18) }
            .joined(separator: "\n")
            + "\n"
        let logData = try #require(content.data(using: .utf8))
        try logData.write(to: logURL)

        RuntimeLaunchLogTrimmer.trimFileIfNeeded(at: logURL, maxRetainedBytes: 160)

        let trimmedContent = try String(contentsOf: logURL, encoding: .utf8)
        let firstLine = try #require(trimmedContent.split(separator: "\n").first.map(String.init))
        let fileSize = try #require(
            (try FileManager.default.attributesOfItem(atPath: logURL.path)[.size] as? NSNumber)?.intValue
        )

        #expect(firstLine.hasPrefix("entry-"))
        #expect(trimmedContent.contains("entry-39-"))
        #expect(!trimmedContent.contains("entry-0-"))
        #expect(fileSize <= 160)
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
    func startingVoiceTurnCreatesDraftBubbleAndWaitsForFirstChunkBeforeOpeningRuntime() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true
        model.composerText = "Typed draft"

        model.startRecording()
        await settleVoiceCallbacks()

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
        #expect(model.voiceRecordingDisplayPhase == .waitingForAudio)
        #expect(model.isVoiceDraftLoading)
        #expect(model.statusText == "Waiting for microphone audio…")
        #expect(model.voiceRecordingHelperText == "Waiting for microphone audio. Press send to finish, or stop to discard the draft.")
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
        #expect(model.isVoiceTranscriptionStarting)
        #expect(model.isVoiceDraftLoading)
        #expect(model.statusText == "Starting transcription…")
        #expect(runtime.sentCommands.isEmpty)

        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        #expect(runtime.sentCommands.map(\.type) == ["start_recording", "append_audio_chunk"])
        #expect(model.lastRuntimeCommand?.type == "append_audio_chunk")
    }

    @Test
    @MainActor
    func recordingStartupStaysLoadingUntilFirstTranscriptDeltaArrives() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        await settleVoiceCallbacks()

        #expect(model.voiceRecordingDisplayPhase == .startingTranscription)
        #expect(model.isVoiceDraftLoading)
        #expect(model.voiceDraftLoadingLabel == "Starting transcription")
        #expect(model.voiceRecordingHelperText == "Starting transcription. Keep speaking and MacAssistant will switch to live text as soon as the first words arrive.")
        #expect(model.statusText == "Starting transcription…")

        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        #expect(model.hasReceivedVoiceTranscript == false)
        #expect(model.voiceRecordingDisplayPhase == .startingTranscription)

        model.handle(event: runtimeEvent(type: "transcript_delta", turnID: turnID, text: "Hello"))

        #expect(model.hasReceivedVoiceTranscript)
        #expect(model.isVoiceDraftLoading == false)
        #expect(model.voiceRecordingDisplayPhase == .transcribing)
        #expect(model.voiceRecordingHelperText == "Listening for speech. Press send to use it, or stop to discard it.")
        #expect(model.statusText == "Listening…")
    }

    @Test
    @MainActor
    func transcriptDeltaUpdatesLiveVoiceDraftCumulatively() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        await settleVoiceCallbacks()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        model.handle(event: runtimeEvent(type: "transcript_delta", turnID: turnID, text: "Hello"))
        model.handle(event: runtimeEvent(type: "transcript_delta", turnID: turnID, text: "Hello there"))

        #expect(currentVoiceDraft(in: model)?.text == "Hello there")
        #expect(model.hasReceivedVoiceTranscript)
        #expect(model.voiceRecordingDisplayPhase == .transcribing)
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
        await settleVoiceCallbacks()
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
        await settleVoiceCallbacks()
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
        await settleVoiceCallbacks()

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
        await settleVoiceCallbacks()

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
        #expect(model.lastRuntimeCommand?.arguments?["stream_reply_speech"] == .bool(true))
    }

    @Test
    @MainActor
    func sendTextCommandIncludesStreamReplySpeechSetting() {
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime)
        model.settings.streamReplySpeechWhileGenerating = false
        model.composerText = "Hello"

        model.sendTextMessage()

        #expect(model.lastRuntimeCommand?.type == "send_text")
        #expect(model.lastRuntimeCommand?.arguments?["stream_reply_speech"] == .bool(false))
    }

    @Test
    @MainActor
    func startRecordingCommandIncludesStreamReplySpeechSetting() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.settings.streamReplySpeechWhileGenerating = false
        model.isMicReady = true

        model.startRecording()
        await settleVoiceCallbacks()
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        let startCommand = try #require(runtime.sentCommands.first(where: { $0.type == "start_recording" }))
        #expect(startCommand.arguments?["stream_reply_speech"] == .bool(false))
    }

    @Test
    @MainActor
    func emptyTranscriptFinalAndFailedTurnFinishedRemoveVoiceDraft() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let runtime = FakeRuntimeClient()
        let model = AppModel(runtime: runtime, microphoneService: microphoneService)
        model.isMicReady = true

        model.startRecording()
        await settleVoiceCallbacks()
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
        await settleVoiceCallbacks()
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
        await settleVoiceCallbacks()

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
        await settleVoiceCallbacks()
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
        #expect(model.statusText == "Starting transcription…")
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
        await settleVoiceCallbacks()
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
        await settleVoiceCallbacks()
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
    func splitRoutePreflightEngagesForBuiltInMicWhileBluetoothOutputIsActive() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let routeCoordinator = FakeMicrophoneRouteCoordinator()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let airPodsOutput = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 301,
            uid: "airpods-output",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        microphoneService.inputDevices = [builtInMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid
        routeCoordinator.defaultOutputDevice = airPodsOutput
        routeCoordinator.preflightResults = [
            .success(.init(defaultOutputDevice: airPodsOutput, didEngageArbitration: true))
        ]
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneRouteCoordinator: routeCoordinator
        )
        model.isMicReady = true
        model.refreshAvailableInputDevices()

        model.startRecording()
        await settleVoiceCallbacks()

        #expect(routeCoordinator.prepareCallCount == 1)
        #expect(routeCoordinator.preparedInputDevices == [builtInMicrophone])
        #expect(microphoneService.startCallCount == 1)
        #expect(model.isRecording)
        #expect(currentVoiceDraft(in: model) != nil)
        #expect(systemStatuses(in: model).isEmpty)
        #expect(model.runtimeLogsText.contains("[Mic] Split-route preflight engaged"))
    }

    @Test
    @MainActor
    func retryableSplitRoutePreflightFailureRetriesSilentlyAndKeepsVoiceDraft() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let routeCoordinator = FakeMicrophoneRouteCoordinator()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let airPodsOutput = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 301,
            uid: "airpods-output",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        microphoneService.inputDevices = [builtInMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid
        routeCoordinator.defaultOutputDevice = airPodsOutput
        routeCoordinator.preflightResults = [
            .failure(MicrophoneRouteCoordinator.PreflightError.arbitrationFailed(message: "Route arbitration failed.")),
            .success(.init(defaultOutputDevice: airPodsOutput, didEngageArbitration: true))
        ]
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneRouteCoordinator: routeCoordinator,
            microphoneStartRetryDelay: .milliseconds(10)
        )
        model.isMicReady = true
        model.refreshAvailableInputDevices()

        model.startRecording()
        await settleVoiceCallbacks()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)

        #expect(microphoneService.startCallCount == 0)
        #expect(currentVoiceDraft(in: model)?.turnID == turnID)
        #expect(systemStatuses(in: model).isEmpty)

        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        #expect(routeCoordinator.prepareCallCount == 2)
        #expect(microphoneService.startCallCount == 1)
        #expect(currentVoiceDraft(in: model)?.turnID == turnID)
        #expect(model.isRecording)
        #expect(systemStatuses(in: model).isEmpty)
    }

    @Test
    @MainActor
    func repeatedRetryableSplitRoutePreflightFailureSurfacesErrorOnce() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let routeCoordinator = FakeMicrophoneRouteCoordinator()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        let airPodsOutput = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 301,
            uid: "airpods-output",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        microphoneService.inputDevices = [builtInMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid
        routeCoordinator.defaultOutputDevice = airPodsOutput
        routeCoordinator.preflightResults = [
            .failure(MicrophoneRouteCoordinator.PreflightError.arbitrationFailed(message: "Route arbitration failed.")),
            .failure(MicrophoneRouteCoordinator.PreflightError.arbitrationFailed(message: "Route arbitration failed."))
        ]
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneRouteCoordinator: routeCoordinator,
            microphoneStartRetryDelay: .milliseconds(10)
        )
        model.isMicReady = true
        model.refreshAvailableInputDevices()

        model.startRecording()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)
        await settleVoiceCallbacks()
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        #expect(routeCoordinator.prepareCallCount == 2)
        #expect(microphoneService.startCallCount == 0)
        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: turnID, in: model) == nil)
        #expect(model.statusText == "Route arbitration failed.")
        #expect(systemStatuses(in: model).map(\.text) == ["Route arbitration failed."])
    }

    @Test
    @MainActor
    func discardingVoiceDraftCancelsPendingSplitRoutePreflight() async throws {
        let microphoneService = FakeMicrophoneCaptureService()
        let routeCoordinator = FakeMicrophoneRouteCoordinator()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        routeCoordinator.defaultOutputDevice = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 301,
            uid: "airpods-output",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        routeCoordinator.holdsPreparationOpen = true
        microphoneService.inputDevices = [builtInMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneRouteCoordinator: routeCoordinator
        )
        model.isMicReady = true
        model.refreshAvailableInputDevices()

        model.startRecording()
        await settleVoiceCallbacks()
        let turnID = try #require(currentVoiceDraft(in: model)?.turnID)

        #expect(microphoneService.startCallCount == 0)
        model.discardCurrentInput()
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()

        #expect(routeCoordinator.cancelPendingPreparationCallCount >= 1)
        #expect(microphoneService.startCallCount == 0)
        #expect(currentVoiceDraft(in: model) == nil)
        #expect(userMessage(for: turnID, in: model) == nil)
        #expect(systemStatuses(in: model).isEmpty)
        #expect(model.statusText == "Ready")
    }

    @Test
    @MainActor
    func bluetoothInputSkipsSplitRoutePreflightArbitration() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let routeCoordinator = FakeMicrophoneRouteCoordinator()
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
        routeCoordinator.defaultOutputDevice = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 301,
            uid: "airpods-output",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        microphoneService.inputDevices = [builtInMicrophone, airPodsMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneRouteCoordinator: routeCoordinator
        )
        model.isMicReady = true
        model.settings.inputDevicePreference = .specificDeviceUID(airPodsMicrophone.uid)
        model.refreshAvailableInputDevices()

        model.startRecording()
        await settleVoiceCallbacks()

        #expect(routeCoordinator.prepareCallCount == 1)
        #expect(microphoneService.lastPreferredInputDeviceUID == airPodsMicrophone.uid)
        #expect(model.runtimeLogsText.contains("[Mic] Split-route preflight skipped requestedInput=My AirPods Pro [airpods-mic]"))
    }

    @Test
    @MainActor
    func microphoneDiagnosticsAppearInDebugLogs() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let routeCoordinator = FakeMicrophoneRouteCoordinator()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        microphoneService.inputDevices = [builtInMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid
        routeCoordinator.defaultOutputDevice = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 301,
            uid: "airpods-output",
            name: "My AirPods Pro",
            transport: .bluetooth
        )
        routeCoordinator.preflightResults = [
            .success(.init(
                defaultOutputDevice: routeCoordinator.defaultOutputDevice,
                didEngageArbitration: true
            ))
        ]
        let runtime = FakeRuntimeClient()
        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneRouteCoordinator: routeCoordinator,
            voiceCaptureWaitingDelay: .milliseconds(10)
        )
        model.isMicReady = true

        model.startRecording()
        try? await Task.sleep(for: .milliseconds(30))
        await settleVoiceCallbacks()
        microphoneService.emitInputBuffer()
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        #expect(model.runtimeLogsText.contains("[Mic] backend=auhal version=test build=test"))
        #expect(model.runtimeLogsText.contains("[Mic] Starting capture"))
        #expect(model.runtimeLogsText.contains("defaultOutput=My AirPods Pro [airpods-output]"))
        #expect(model.runtimeLogsText.contains("[Mic] Split-route preflight engaged"))
        #expect(model.runtimeLogsText.contains("[Mic] Capture session started"))
        #expect(model.runtimeLogsText.contains("[Mic] Waiting for microphone audio"))
        #expect(model.runtimeLogsText.contains("[Mic] First input buffer"))
        #expect(model.runtimeLogsText.contains("[Mic] First converted chunk"))
    }

    @Test
    @MainActor
    func automaticInputDevicePreferencePrefersBuiltInMicrophoneOverAirPods() async {
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
        await settleVoiceCallbacks()

        #expect(microphoneService.lastPreferredInputDeviceUID == builtInMicrophone.uid)
        #expect(model.runtimeLogsText.contains("resolved=MacBook Pro Microphone [built-in-mic]"))
    }

    @Test
    @MainActor
    func explicitInputDevicePreferencePassesSelectedUIDToCaptureService() async {
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
        await settleVoiceCallbacks()

        #expect(microphoneService.lastPreferredInputDeviceUID == airPodsMicrophone.uid)
        #expect(model.runtimeLogsText.contains("specific My AirPods Pro [airpods-mic]"))
    }

    @Test
    @MainActor
    func captureStartLogsSelectedInputDeviceAndFormat() async {
        let microphoneService = FakeMicrophoneCaptureService()
        let routeCoordinator = FakeMicrophoneRouteCoordinator()
        let runtime = FakeRuntimeClient()
        let builtInMicrophone = MicrophoneCaptureService.InputDevice(
            deviceID: 101,
            uid: "built-in-mic",
            name: "MacBook Pro Microphone",
            transport: .builtIn
        )
        microphoneService.inputDevices = [builtInMicrophone]
        microphoneService.defaultInputDeviceUID = builtInMicrophone.uid
        routeCoordinator.defaultOutputDevice = MicrophoneRouteCoordinator.OutputDevice(
            deviceID: 302,
            uid: "built-in-output",
            name: "MacBook Pro Speakers",
            transport: .builtIn
        )

        let model = AppModel(
            runtime: runtime,
            microphoneService: microphoneService,
            microphoneRouteCoordinator: routeCoordinator
        )
        model.isMicReady = true
        model.refreshAvailableInputDevices()

        model.startRecording()
        await settleVoiceCallbacks()
        microphoneService.emitChunk()
        await settleVoiceCallbacks()

        #expect(model.runtimeLogsText.contains("[Mic] Starting capture preference=automatic (built-in preferred) resolved=MacBook Pro Microphone [built-in-mic] defaultOutput=MacBook Pro Speakers [built-in-output]"))
        #expect(model.runtimeLogsText.contains("[Mic] Split-route preflight skipped requestedInput=MacBook Pro Microphone [built-in-mic] defaultOutput=MacBook Pro Speakers [built-in-output]"))
        #expect(model.runtimeLogsText.contains("[Mic] Capture session started device=MacBook Pro Microphone [built-in-mic]"))
        #expect(model.runtimeLogsText.contains("selected=MacBook Pro Microphone [built-in-mic]"))
        #expect(model.runtimeLogsText.contains("[Mic] First converted chunk"))
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

        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: turnID,
            text: "Hello there",
            assistantSegmentID: "segment-final",
            isFinalSegment: true
        ))
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
            type: "assistant_delta",
            turnID: turnID,
            text: "Final answer",
            assistantSegmentID: "segment-final",
            isFinalSegment: true
        ))
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

        model.handle(event: runtimeEvent(
            type: "assistant_delta",
            turnID: turnID,
            text: "Finished answer",
            assistantSegmentID: "segment-final",
            isFinalSegment: true
        ))
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
        assistantSegmentID: String? = nil,
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
        safetyClass: String? = nil,
        approvalState: String? = nil,
        executionState: String? = nil,
        output: String? = nil,
        status: String? = nil,
        isFinal: Bool? = nil,
        isFinalSegment: Bool? = nil
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
            assistantSegmentID: assistantSegmentID,
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
            safetyClass: safetyClass,
            approvalState: approvalState,
            executionState: executionState,
            output: output,
            status: status,
            isFinal: isFinal,
            isFinalSegment: isFinalSegment
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
        let messages = model.conversation.compactMap { item -> AssistantMessage? in
            guard case .assistantMessage(let message) = item,
                  message.turnID == turnID else { return nil }
            return message
        }
        return messages.last(where: \.isFinalSegment) ?? messages.last
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
        for _ in 0..<6 {
            await Task.yield()
        }
        try? await Task.sleep(for: .milliseconds(1))
        for _ in 0..<2 {
            await Task.yield()
        }
    }
}

private final class FakeMicrophoneCaptureService: MicrophoneCaptureServicing, @unchecked Sendable {
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
                logHandler?("[Mic] backend=auhal version=test build=test")
                logHandler?(
                    "[Mic] Failed to start capture requested=\(describe(device: inputDevices.first(where: { $0.uid == preferredInputDeviceUID }))) " +
                    "selected=\(describe(device: lastStartedInputDevice)) " +
                    "device=\(describe(device: lastStartedInputDevice)) " +
                    "defaultInput=\(describe(device: defaultInputDevice())) " +
                    "defaultOutput=system-default splitRouteArbitration=skipped " +
                    "deviceFormat=44100Hz/1ch/float32 clientFormat=44100Hz/1ch/float32: \(error.localizedDescription)"
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
        logHandler?("[Mic] backend=auhal version=test build=test")
        logHandler?(
            "[Mic] Capture session started device=\(describe(device: lastStartedInputDevice)) " +
            "selected=\(describe(device: lastStartedInputDevice)) " +
            "requested=\(describe(device: inputDevices.first(where: { $0.uid == preferredInputDeviceUID }))) " +
            "defaultInput=\(describe(device: defaultInputDevice())) " +
            "defaultOutput=system-default splitRouteArbitration=skipped " +
            "deviceFormat=44100Hz/1ch/float32 clientFormat=44100Hz/1ch/float32"
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

private final class FakeMicrophoneRouteCoordinator: MicrophoneRouteCoordinating, @unchecked Sendable {
    var defaultOutputDevice: MicrophoneRouteCoordinator.OutputDevice?
    var preflightResults: [Result<MicrophoneRouteCoordinator.PreflightResult, Error>] = []
    var holdsPreparationOpen = false
    private(set) var prepareCallCount = 0
    private(set) var cancelPendingPreparationCallCount = 0
    private(set) var leaveRecordingRouteCallCount = 0
    private(set) var preparedInputDevices: [MicrophoneCaptureService.InputDevice?] = []
    private var pendingContinuations: [CheckedContinuation<MicrophoneRouteCoordinator.PreflightResult, Error>] = []
    private var arbitrationActive = false

    func prepareForRecording(inputDevice: MicrophoneCaptureService.InputDevice?) async throws -> MicrophoneRouteCoordinator.PreflightResult {
        prepareCallCount += 1
        preparedInputDevices.append(inputDevice)

        if holdsPreparationOpen {
            return try await withCheckedThrowingContinuation { continuation in
                pendingContinuations.append(continuation)
            }
        }

        if !preflightResults.isEmpty {
            let result = preflightResults.removeFirst()
            switch result {
            case .success(let preflightResult):
                defaultOutputDevice = preflightResult.defaultOutputDevice
                arbitrationActive = preflightResult.didEngageArbitration
                return preflightResult
            case .failure(let error):
                throw error
            }
        }

        arbitrationActive = false
        return MicrophoneRouteCoordinator.PreflightResult(
            defaultOutputDevice: defaultOutputDevice,
            didEngageArbitration: false
        )
    }

    func cancelPendingPreparation() {
        cancelPendingPreparationCallCount += 1
        let continuations = pendingContinuations
        pendingContinuations.removeAll(keepingCapacity: false)
        continuations.forEach { $0.resume(throwing: CancellationError()) }
    }

    func leaveRecordingRoute() {
        leaveRecordingRouteCallCount += 1
        arbitrationActive = false
    }

    func routeStateSnapshot() -> MicrophoneRouteCoordinator.RouteState {
        MicrophoneRouteCoordinator.RouteState(
            defaultOutputDevice: defaultOutputDevice,
            isArbitrationActive: arbitrationActive
        )
    }

    func resolvePendingPreparation(with result: Result<MicrophoneRouteCoordinator.PreflightResult, Error>) {
        let continuations = pendingContinuations
        pendingContinuations.removeAll(keepingCapacity: false)
        switch result {
        case .success(let preflightResult):
            defaultOutputDevice = preflightResult.defaultOutputDevice
            arbitrationActive = preflightResult.didEngageArbitration
        case .failure:
            arbitrationActive = false
        }
        continuations.forEach { continuation in
            switch result {
            case .success(let preflightResult):
                continuation.resume(returning: preflightResult)
            case .failure(let error):
                continuation.resume(throwing: error)
            }
        }
    }
}

private final class FakeCaptureControllerFactory: MicrophoneCaptureControllerFactory, @unchecked Sendable {
    var queuedControllers: [FakeCaptureController] = []
    private(set) var selectedInputDeviceUIDs: [String?] = []

    func makeController(
        selectedInputDevice: MicrophoneCaptureService.InputDevice?,
        onPCMBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) throws -> any MicrophoneCaptureControlling {
        guard !queuedControllers.isEmpty else {
            throw NSError(
                domain: "MacAssistantTests.Microphone",
                code: 500,
                userInfo: [NSLocalizedDescriptionKey: "No fake capture controller was queued."]
            )
        }

        let controller = queuedControllers.removeFirst()
        controller.selectedInputDevice = selectedInputDevice
        controller.onPCMBuffer = onPCMBuffer
        controller.onError = onError
        selectedInputDeviceUIDs.append(selectedInputDevice?.uid)
        return controller
    }
}

private final class FakeCaptureController: MicrophoneCaptureControlling, @unchecked Sendable {
    var selectedInputDevice: MicrophoneCaptureService.InputDevice?
    let boundInputDeviceID: AudioDeviceID?
    let boundInputDeviceUID: String?
    let boundInputDeviceName: String?
    let deviceInputFormat: AVAudioFormat?
    let clientInputFormat: AVAudioFormat?
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0
    var startError: Error?
    var onPCMBuffer: (@Sendable (AVAudioPCMBuffer) -> Void)?
    var onError: (@Sendable (Error) -> Void)?

    init(
        boundInputDeviceID: AudioDeviceID?,
        boundInputDeviceUID: String?,
        boundInputDeviceName: String?,
        deviceInputFormat: AVAudioFormat?,
        clientInputFormat: AVAudioFormat?
    ) {
        self.boundInputDeviceID = boundInputDeviceID
        self.boundInputDeviceUID = boundInputDeviceUID
        self.boundInputDeviceName = boundInputDeviceName
        self.deviceInputFormat = deviceInputFormat
        self.clientInputFormat = clientInputFormat
    }

    func start() throws {
        startCallCount += 1
        if let startError {
            throw startError
        }
    }

    func stop() {
        stopCallCount += 1
    }

    func emit(buffer: AVAudioPCMBuffer) {
        onPCMBuffer?(buffer)
    }

    func emitError(_ error: Error) {
        onError?(error)
    }
}

private final class FakeAUHALInputUnit: AUHALInputUnitControlling, @unchecked Sendable {
    private(set) var setInputEnabledValues: [Bool] = []
    private(set) var setOutputEnabledValues: [Bool] = []
    private(set) var setCurrentDeviceValues: [AudioDeviceID] = []
    private(set) var setClientInputFormats: [AudioStreamBasicDescription] = []
    private(set) var initializeCallCount = 0
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0
    private(set) var uninitializeCallCount = 0
    var currentDeviceIDValue: AudioDeviceID
    var inputDeviceFormatValue: AudioStreamBasicDescription
    var startError: Error?
    var renderError: Error?
    private var inputCallback: AURenderCallbackStruct?

    init(currentDeviceIDValue: AudioDeviceID, inputDeviceFormatValue: AudioStreamBasicDescription) {
        self.currentDeviceIDValue = currentDeviceIDValue
        self.inputDeviceFormatValue = inputDeviceFormatValue
    }

    func setInputEnabled(_ enabled: Bool) throws {
        setInputEnabledValues.append(enabled)
    }

    func setOutputEnabled(_ enabled: Bool) throws {
        setOutputEnabledValues.append(enabled)
    }

    func setCurrentDevice(_ deviceID: AudioDeviceID) throws {
        setCurrentDeviceValues.append(deviceID)
        currentDeviceIDValue = deviceID
    }

    func currentDeviceID() throws -> AudioDeviceID {
        currentDeviceIDValue
    }

    func inputDeviceFormat() throws -> AudioStreamBasicDescription {
        inputDeviceFormatValue
    }

    func setClientInputFormat(_ format: AudioStreamBasicDescription) throws {
        setClientInputFormats.append(format)
    }

    func setInputCallback(_ callback: AURenderCallbackStruct) throws {
        inputCallback = callback
    }

    func initialize() throws {
        initializeCallCount += 1
    }

    func start() throws {
        startCallCount += 1
        if let startError {
            throw startError
        }
    }

    func render(
        ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
        timeStamp: UnsafePointer<AudioTimeStamp>,
        busNumber: UInt32,
        frameCount: UInt32,
        ioData: UnsafeMutablePointer<AudioBufferList>
    ) throws {
        if let renderError {
            throw renderError
        }

        let audioBuffers = UnsafeMutableAudioBufferListPointer(ioData)
        let format = AVAudioFormat(streamDescription: withUnsafePointer(to: inputDeviceFormatValue) { pointer in
            UnsafeMutablePointer(mutating: pointer)
        })!
        let bytesPerFrame = Int(format.streamDescription.pointee.mBytesPerFrame)
        let byteCount = Int(frameCount) * max(bytesPerFrame, MemoryLayout<Float>.size)

        for index in audioBuffers.indices {
            guard let data = audioBuffers[index].mData else { continue }
            memset(data, 0, byteCount)
            audioBuffers[index].mDataByteSize = UInt32(byteCount)
        }
    }

    func stop() {
        stopCallCount += 1
    }

    func uninitialize() {
        uninitializeCallCount += 1
    }

    func invokeInputCallback(frameCount: UInt32) {
        guard
            let callback = inputCallback,
            let inputProc = callback.inputProc,
            let refCon = callback.inputProcRefCon
        else {
            Issue.record("Expected AUHAL input callback to be configured.")
            return
        }
        var flags = AudioUnitRenderActionFlags()
        var timestamp = AudioTimeStamp()
        _ = inputProc(
            refCon,
            &flags,
            &timestamp,
            1,
            frameCount,
            nil
        )
    }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func increment() {
        lock.lock()
        storage += 1
        lock.unlock()
    }
}

private final class LockedLogMessages: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    var values: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func append(_ message: String) {
        lock.lock()
        storage.append(message)
        lock.unlock()
    }

    func contains(_ fragment: String) -> Bool {
        values.contains(where: { $0.contains(fragment) })
    }
}

private final class LockedChunkRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [MicrophoneCaptureService.CaptureChunk] = []

    var chunks: [MicrophoneCaptureService.CaptureChunk] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func append(_ chunk: MicrophoneCaptureService.CaptureChunk) {
        lock.lock()
        storage.append(chunk)
        lock.unlock()
    }
}

private final class LockedPCMBufferRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [AVAudioPCMBuffer] = []

    var buffers: [AVAudioPCMBuffer] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        storage.append(buffer)
        lock.unlock()
    }
}

private func makeFloatPCMBuffer(
    sampleRate: Double,
    frameCount: AVAudioFrameCount,
    channels: AVAudioChannelCount = 1
) -> AVAudioPCMBuffer {
    let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: sampleRate,
        channels: channels,
        interleaved: false
    )!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
    buffer.frameLength = frameCount

    if let floatChannelData = buffer.floatChannelData {
        for channel in 0..<Int(channels) {
            for frame in 0..<Int(frameCount) {
                floatChannelData[channel][frame] = sin(Float(frame) * 0.05)
            }
        }
    }

    return buffer
}

private func makeFloatStreamDescription(
    sampleRate: Double,
    channels: AVAudioChannelCount
) -> AudioStreamBasicDescription {
    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: channels)!
    return format.streamDescription.pointee
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
