@preconcurrency import AVFoundation
import AudioToolbox
import CoreAudio
import CoreMedia
import Foundation

protocol MicrophoneCaptureServicing: AnyObject {
    var onInputDevicesChanged: (@Sendable () -> Void)? { get set }

    func authorizationStatus() -> MicrophoneCaptureService.AuthorizationStatus
    func requestAccess() async -> Bool
    func availableInputDevices() -> [MicrophoneCaptureService.InputDevice]
    func defaultInputDevice() -> MicrophoneCaptureService.InputDevice?
    func start(
        preferredInputDeviceUID: String?,
        onStarted: (@Sendable () -> Void)?,
        onFirstInputBuffer: (@Sendable () -> Void)?,
        onFirstChunk: (@Sendable () -> Void)?,
        onError: (@Sendable (Error) -> Void)?,
        logHandler: (@Sendable (String) -> Void)?,
        chunkHandler: @escaping @Sendable (MicrophoneCaptureService.CaptureChunk) -> Void
    ) throws
    func stop()
}

protocol MicrophoneCaptureControlling: AnyObject, Sendable {
    var requestedInputDeviceUID: String? { get }
    var resolvedInputDeviceUID: String? { get }
    var resolvedInputDeviceName: String? { get }

    func start() throws
    func stop()
}

protocol MicrophoneCaptureControllerFactory: Sendable {
    func makeController(
        preferredInputDeviceUID: String?,
        onPCMBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) throws -> any MicrophoneCaptureControlling
}

final class MicrophoneCaptureService: NSObject, MicrophoneCaptureServicing, @unchecked Sendable {
    struct CaptureChunk: Sendable {
        let data: Data
        let sampleRate: Int
        let channels: Int
    }

    struct InputDevice: Identifiable, Equatable, Sendable {
        enum Transport: String, Sendable {
            case builtIn
            case bluetooth
            case usb
            case aggregate
            case virtual
            case unknown
        }

        let deviceID: AudioDeviceID
        let uid: String
        let name: String
        let transport: Transport

        var id: String { uid }
        var isBuiltIn: Bool { transport == .builtIn }
        var isBluetooth: Bool { transport == .bluetooth }
    }

    enum AuthorizationStatus {
        case notDetermined
        case denied
        case authorized
    }

    enum CaptureError: LocalizedError {
        case permissionDenied
        case missingInputFormat
        case missingCaptureInputDevice
        case unableToBuildOutputFormat
        case unableToCreateConverter
        case unableToCreateConvertedBuffer
        case unableToCreateCaptureInput(name: String)
        case unableToAddCaptureInput(name: String)
        case unableToAddCaptureOutput
        case captureSessionFailedToStart
        case captureSessionInterrupted(reason: String)
        case unsupportedSampleBufferFormat
        case unableToCreatePCMBuffer

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Microphone access is required to record voice input."
            case .missingInputFormat:
                return "No microphone input format is available."
            case .missingCaptureInputDevice:
                return "No microphone input device is currently available."
            case .unableToBuildOutputFormat:
                return "Unable to create the voice capture output format."
            case .unableToCreateConverter:
                return "Unable to convert microphone audio into the speech input format."
            case .unableToCreateConvertedBuffer:
                return "Unable to allocate the converted microphone audio buffer."
            case .unableToCreateCaptureInput(let name):
                return "Unable to open microphone input device \"\(name)\"."
            case .unableToAddCaptureInput(let name):
                return "Unable to attach microphone input device \"\(name)\" to the capture session."
            case .unableToAddCaptureOutput:
                return "Unable to attach microphone capture output."
            case .captureSessionFailedToStart:
                return "Unable to start microphone capture."
            case .captureSessionInterrupted(let reason):
                return "Microphone capture was interrupted (\(reason))."
            case .unsupportedSampleBufferFormat:
                return "The microphone produced an unsupported audio format."
            case .unableToCreatePCMBuffer:
                return "Unable to read microphone audio samples."
            }
        }
    }

    static let targetSampleRate = 16_000.0
    static let targetChannels: AVAudioChannelCount = 1
    static let tapBufferSize: AVAudioFrameCount = 1_024

    var onInputDevicesChanged: (@Sendable () -> Void)?

    private let stateLock = NSLock()
    private let audioDeviceNotificationQueue = DispatchQueue(label: "MacAssistant.MicrophoneCaptureService.devices")
    private let authorizationStatusProvider: @Sendable () -> AuthorizationStatus
    private let requestAccessHandler: @Sendable () async -> Bool
    private let availableInputDevicesProvider: @Sendable () -> [InputDevice]
    private let defaultInputDeviceProvider: @Sendable () -> InputDevice?
    private let routeStateProvider: @Sendable () -> MicrophoneRouteCoordinator.RouteState
    private let buildInfoProvider: @Sendable () -> (version: String, build: String)
    private let captureControllerFactory: any MicrophoneCaptureControllerFactory
    private var activeSession: CaptureSession?
    private static let backendBannerLock = NSLock()
    nonisolated(unsafe) private static var hasLoggedBackendBanner = false

    private lazy var devicesChangedListener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
        self?.onInputDevicesChanged?()
    }

    private lazy var defaultInputChangedListener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
        self?.onInputDevicesChanged?()
    }

    init(
        authorizationStatusProvider: @escaping @Sendable () -> AuthorizationStatus = {
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized:
                return .authorized
            case .notDetermined:
                return .notDetermined
            case .denied, .restricted:
                return .denied
            @unknown default:
                return .denied
            }
        },
        requestAccessHandler: @escaping @Sendable () async -> Bool = {
            await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        },
        availableInputDevicesProvider: @escaping @Sendable () -> [InputDevice] = { MicrophoneCaptureService.enumerateInputDevices() },
        defaultInputDeviceProvider: @escaping @Sendable () -> InputDevice? = { MicrophoneCaptureService.resolveDefaultInputDevice() },
        routeStateProvider: @escaping @Sendable () -> MicrophoneRouteCoordinator.RouteState = {
            MicrophoneRouteCoordinator.shared.routeStateSnapshot()
        },
        buildInfoProvider: @escaping @Sendable () -> (version: String, build: String) = {
            let infoDictionary = Bundle.main.infoDictionary ?? [:]
            let version = infoDictionary["CFBundleShortVersionString"] as? String ?? "unknown"
            let build = infoDictionary["CFBundleVersion"] as? String ?? "unknown"
            return (version, build)
        },
        captureControllerFactory: any MicrophoneCaptureControllerFactory = AVFoundationMicrophoneCaptureControllerFactory()
    ) {
        self.authorizationStatusProvider = authorizationStatusProvider
        self.requestAccessHandler = requestAccessHandler
        self.availableInputDevicesProvider = availableInputDevicesProvider
        self.defaultInputDeviceProvider = defaultInputDeviceProvider
        self.routeStateProvider = routeStateProvider
        self.buildInfoProvider = buildInfoProvider
        self.captureControllerFactory = captureControllerFactory
        super.init()
        installAudioDeviceListeners()
    }

    deinit {
        removeAudioDeviceListeners()
    }

    func authorizationStatus() -> AuthorizationStatus {
        authorizationStatusProvider()
    }

    func requestAccess() async -> Bool {
        let status = authorizationStatus()
        if status == .authorized {
            return true
        }
        if status == .denied {
            return false
        }

        return await requestAccessHandler()
    }

    func availableInputDevices() -> [InputDevice] {
        availableInputDevicesProvider()
    }

    func defaultInputDevice() -> InputDevice? {
        defaultInputDeviceProvider()
    }

    func start(
        preferredInputDeviceUID: String? = nil,
        onStarted: (@Sendable () -> Void)? = nil,
        onFirstInputBuffer: (@Sendable () -> Void)? = nil,
        onFirstChunk: (@Sendable () -> Void)? = nil,
        onError: (@Sendable (Error) -> Void)? = nil,
        logHandler: (@Sendable (String) -> Void)? = nil,
        chunkHandler: @escaping @Sendable (CaptureChunk) -> Void
    ) throws {
        guard authorizationStatus() == .authorized else {
            throw CaptureError.permissionDenied
        }
        guard currentSession() == nil else { return }

        if let backendBanner = Self.backendBanner(buildInfoProvider: buildInfoProvider) {
            logHandler?(backendBanner)
        }

        let availableDevices = availableInputDevices()
        let requestedInputDevice = preferredInputDeviceUID.flatMap { uid in
            availableDevices.first(where: { $0.uid == uid })
        }
        if let preferredInputDeviceUID, requestedInputDevice == nil {
            logHandler?("[Mic] Requested input device uid=\(preferredInputDeviceUID) is unavailable. Falling back to the active system route.")
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Self.targetSampleRate,
            channels: Self.targetChannels,
            interleaved: true
        ) else {
            throw CaptureError.unableToBuildOutputFormat
        }

        let currentDefaultInputDevice = defaultInputDevice()
        let routeState = routeStateProvider()
        let fallbackInputDevice = requestedInputDevice ?? currentDefaultInputDevice ?? availableDevices.first
        let session = CaptureSession(
            outputFormat: outputFormat,
            requestedInputDevice: requestedInputDevice,
            defaultInputDevice: currentDefaultInputDevice,
            defaultOutputDevice: routeState.defaultOutputDevice,
            didEngageSplitRouteArbitration: routeState.isArbitrationActive,
            resolvedInputDeviceUID: fallbackInputDevice?.uid,
            resolvedInputDeviceName: fallbackInputDevice?.name,
            chunkHandler: chunkHandler,
            onStarted: onStarted,
            onFirstInputBuffer: onFirstInputBuffer,
            onFirstChunk: onFirstChunk,
            onError: onError,
            logHandler: logHandler
        )

        let controller = try captureControllerFactory.makeController(
            preferredInputDeviceUID: requestedInputDevice?.uid,
            onPCMBuffer: { [weak self] buffer in
                self?.handlePCMBuffer(buffer, session: session)
            },
            onError: { [weak self] error in
                self?.emitError(error, session: session)
            }
        )
        session.controller = controller
        setSession(session)

        do {
            try controller.start()
        } catch {
            clearSessionIfNeeded(session)
            teardown(session: session)
            session.logHandler?(
                "[Mic] Failed to start capture requested=\(Self.describe(device: session.requestedInputDevice)) " +
                "resolved=\(Self.describe(name: session.resolvedInputDeviceName, uid: session.resolvedInputDeviceUID)) " +
                "defaultInput=\(Self.describe(device: session.defaultInputDevice)) " +
                "defaultOutput=\(Self.describe(outputDevice: session.defaultOutputDevice)) " +
                "splitRouteArbitration=\(Self.describe(splitRouteArbitration: session.didEngageSplitRouteArbitration)): \(error.localizedDescription)"
            )
            throw error
        }

        session.resolvedInputDeviceUID = controller.resolvedInputDeviceUID ?? session.resolvedInputDeviceUID
        session.resolvedInputDeviceName = controller.resolvedInputDeviceName ?? session.resolvedInputDeviceName
        session.logHandler?(
            "[Mic] Capture session started device=\(Self.describe(name: session.resolvedInputDeviceName, uid: session.resolvedInputDeviceUID)) " +
            "requested=\(Self.describe(device: session.requestedInputDevice)) " +
            "defaultInput=\(Self.describe(device: session.defaultInputDevice)) " +
            "defaultOutput=\(Self.describe(outputDevice: session.defaultOutputDevice)) " +
            "splitRouteArbitration=\(Self.describe(splitRouteArbitration: session.didEngageSplitRouteArbitration))"
        )
        session.onStarted?()
    }

    func stop() {
        guard let session = takeCurrentSession() else { return }
        teardown(session: session)
        session.logHandler?(
            "[Mic] Stopped capture after \(Self.elapsedDescription(since: session.startedAt)) " +
            "(device=\(Self.describe(name: session.resolvedInputDeviceName, uid: session.resolvedInputDeviceUID)), rawInput=\(session.hasDeliveredFirstInputBuffer), convertedChunk=\(session.hasDeliveredFirstChunk))"
        )
    }

    fileprivate static func makePCMBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard
            CMSampleBufferGetNumSamples(sampleBuffer) > 0,
            let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
            let streamDescriptionPointer = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)
        else {
            return nil
        }

        var streamDescription = streamDescriptionPointer.pointee
        guard let format = AVAudioFormat(streamDescription: &streamDescription) else {
            return nil
        }

        let frameLength = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameLength) else {
            return nil
        }
        pcmBuffer.frameLength = frameLength

        var bufferListSizeNeeded = 0
        let sizingStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &bufferListSizeNeeded,
            bufferListOut: nil,
            bufferListSize: 0,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: nil
        )
        guard sizingStatus == noErr, bufferListSizeNeeded > 0 else {
            return nil
        }

        let rawBufferList = UnsafeMutableRawPointer.allocate(
            byteCount: bufferListSizeNeeded,
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { rawBufferList.deallocate() }

        let audioBufferList = rawBufferList.bindMemory(to: AudioBufferList.self, capacity: 1)
        var retainedBlockBuffer: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &bufferListSizeNeeded,
            bufferListOut: audioBufferList,
            bufferListSize: bufferListSizeNeeded,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &retainedBlockBuffer
        )
        guard status == noErr else {
            return nil
        }

        copyAudioBufferList(audioBufferList, into: pcmBuffer)
        return pcmBuffer
    }

    private func handlePCMBuffer(_ buffer: AVAudioPCMBuffer, session: CaptureSession) {
        guard isCurrentSession(session) else { return }
        guard buffer.format.channelCount > 0 else {
            emitError(CaptureError.missingInputFormat, session: session)
            return
        }

        if markFirstInputBufferIfNeeded(for: session) {
            session.logHandler?(
                "[Mic] First input buffer after \(Self.elapsedDescription(since: session.startedAt)) " +
                "device=\(Self.describe(name: session.resolvedInputDeviceName, uid: session.resolvedInputDeviceUID)) " +
                "format=\(Self.describe(format: buffer.format))"
            )
            session.onFirstInputBuffer?()
        }

        guard let converter = ensureConverter(for: buffer.format, session: session) else { return }

        let outputCapacity = Self.convertedFrameCapacity(
            inputFrameLength: buffer.frameLength,
            inputSampleRate: buffer.format.sampleRate,
            outputSampleRate: session.outputFormat.sampleRate
        )
        guard let converted = AVAudioPCMBuffer(pcmFormat: session.outputFormat, frameCapacity: outputCapacity) else {
            emitError(CaptureError.unableToCreateConvertedBuffer, session: session)
            return
        }

        var error: NSError?
        let inputBlock = Self.makeSingleUseInputBlock(buffer: buffer)
        let status = converter.convert(to: converted, error: &error, withInputFrom: inputBlock)
        if let error {
            emitError(error, session: session)
            return
        }
        guard status != .error else { return }

        guard
            let channelData = converted.int16ChannelData,
            converted.frameLength > 0
        else { return }

        let frameLength = Int(converted.frameLength)
        let byteCount = frameLength * MemoryLayout<Int16>.size
        let data = Data(bytes: channelData[0], count: byteCount)
        #if DEBUG
        let inputDuration = Double(buffer.frameLength) / buffer.format.sampleRate
        let outputDuration = Double(converted.frameLength) / session.outputFormat.sampleRate
        if abs(inputDuration - outputDuration) > 0.05 {
            NSLog(
                "Microphone conversion drift detected: input=%.4fs output=%.4fs",
                inputDuration,
                outputDuration
            )
        }
        #endif

        if markFirstChunkIfNeeded(for: session) {
            session.logHandler?(
                "[Mic] First converted chunk after \(Self.elapsedDescription(since: session.startedAt)) " +
                "device=\(Self.describe(name: session.resolvedInputDeviceName, uid: session.resolvedInputDeviceUID)) " +
                "input=\(Self.describe(format: buffer.format)) output=\(Self.describe(format: session.outputFormat))"
            )
            session.onFirstChunk?()
        }
        session.chunkHandler(CaptureChunk(
            data: data,
            sampleRate: Int(Self.targetSampleRate),
            channels: Int(Self.targetChannels)
        ))
    }

    private func ensureConverter(for inputFormat: AVAudioFormat, session: CaptureSession) -> AVAudioConverter? {
        if let currentFormat = session.inputFormat,
           Self.formatsMatch(currentFormat, inputFormat),
           let converter = session.converter {
            return converter
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: session.outputFormat) else {
            emitError(CaptureError.unableToCreateConverter, session: session)
            return nil
        }

        let previousInputFormat = session.inputFormat
        session.inputFormat = inputFormat
        session.converter = converter

        if previousInputFormat == nil {
            session.logHandler?("[Mic] Active input format=\(Self.describe(format: inputFormat)); converter ready.")
        } else if let previousInputFormat, !Self.formatsMatch(previousInputFormat, inputFormat) {
            session.logHandler?(
                "[Mic] Input format changed to \(Self.describe(format: inputFormat)); rebuilding converter."
            )
        }

        return converter
    }

    private func currentSession() -> CaptureSession? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return activeSession
    }

    private func setSession(_ session: CaptureSession) {
        stateLock.lock()
        activeSession = session
        stateLock.unlock()
    }

    private func takeCurrentSession() -> CaptureSession? {
        stateLock.lock()
        defer { stateLock.unlock() }
        let session = activeSession
        activeSession = nil
        return session
    }

    private func isCurrentSession(_ session: CaptureSession) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return activeSession === session
    }

    private func clearSessionIfNeeded(_ session: CaptureSession) {
        stateLock.lock()
        if activeSession === session {
            activeSession = nil
        }
        stateLock.unlock()
    }

    private func markFirstChunkIfNeeded(for session: CaptureSession) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard activeSession === session, !session.hasDeliveredFirstChunk else { return false }
        session.hasDeliveredFirstChunk = true
        return true
    }

    private func markFirstInputBufferIfNeeded(for session: CaptureSession) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard activeSession === session, !session.hasDeliveredFirstInputBuffer else { return false }
        session.hasDeliveredFirstInputBuffer = true
        return true
    }

    private func emitError(_ error: Error, session: CaptureSession) {
        guard isCurrentSession(session) else { return }
        session.logHandler?("[Mic] Capture error after \(Self.elapsedDescription(since: session.startedAt)): \(error.localizedDescription)")
        session.onError?(error)
    }

    private func teardown(session: CaptureSession) {
        session.controller?.stop()
        session.controller = nil
        session.converter = nil
        session.inputFormat = nil
    }

    private func installAudioDeviceListeners() {
        var devicesPropertyAddress = Self.devicesPropertyAddress()
        let devicesStatus = AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &devicesPropertyAddress,
            audioDeviceNotificationQueue,
            devicesChangedListener
        )
        if devicesStatus != noErr {
            NSLog("Failed to listen for audio device changes: %d", devicesStatus)
        }

        var defaultInputPropertyAddress = Self.defaultInputPropertyAddress()
        let defaultInputStatus = AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &defaultInputPropertyAddress,
            audioDeviceNotificationQueue,
            defaultInputChangedListener
        )
        if defaultInputStatus != noErr {
            NSLog("Failed to listen for default input changes: %d", defaultInputStatus)
        }
    }

    private func removeAudioDeviceListeners() {
        var devicesPropertyAddress = Self.devicesPropertyAddress()
        AudioObjectRemovePropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &devicesPropertyAddress,
            audioDeviceNotificationQueue,
            devicesChangedListener
        )
        var defaultInputPropertyAddress = Self.defaultInputPropertyAddress()
        AudioObjectRemovePropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &defaultInputPropertyAddress,
            audioDeviceNotificationQueue,
            defaultInputChangedListener
        )
    }

    private static func devicesPropertyAddress() -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    private static func defaultInputPropertyAddress() -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    private static func enumerateInputDevices() -> [InputDevice] {
        var devicesAddress = devicesPropertyAddress()
        guard let deviceIDs: [AudioDeviceID] = readArrayProperty(
            objectID: AudioObjectID(kAudioObjectSystemObject),
            address: &devicesAddress
        ) else {
            return []
        }

        return deviceIDs.compactMap { deviceID in
            guard hasInputStreams(deviceID: deviceID) else { return nil }
            let uid = deviceUID(for: deviceID) ?? "device-\(deviceID)"
            let name = deviceName(for: deviceID) ?? uid
            return InputDevice(
                deviceID: deviceID,
                uid: uid,
                name: name,
                transport: transportType(for: deviceID)
            )
        }
        .sorted { lhs, rhs in
            if lhs.isBuiltIn != rhs.isBuiltIn {
                return lhs.isBuiltIn && !rhs.isBuiltIn
            }
            if lhs.isBluetooth != rhs.isBluetooth {
                return !lhs.isBluetooth && rhs.isBluetooth
            }
            return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }
    }

    private static func resolveDefaultInputDevice() -> InputDevice? {
        guard let defaultDeviceID = defaultInputDeviceID() else { return nil }
        return enumerateInputDevices().first(where: { $0.deviceID == defaultDeviceID })
    }

    private static func defaultInputDeviceID() -> AudioDeviceID? {
        var defaultInputAddress = defaultInputPropertyAddress()
        return readScalarProperty(
            objectID: AudioObjectID(kAudioObjectSystemObject),
            address: &defaultInputAddress,
            as: AudioDeviceID.self
        )
    }

    private static func hasInputStreams(deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectHasProperty(deviceID, &address) else { return false }

        var propertySize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &propertySize) == noErr else { return false }
        guard propertySize >= UInt32(MemoryLayout<AudioBufferList>.size) else { return false }

        let rawBufferList = UnsafeMutableRawPointer.allocate(
            byteCount: Int(propertySize),
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { rawBufferList.deallocate() }

        let bufferList = rawBufferList.bindMemory(to: AudioBufferList.self, capacity: 1)
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &propertySize, bufferList) == noErr else {
            return false
        }

        return UnsafeMutableAudioBufferListPointer(bufferList).contains { buffer in
            buffer.mNumberChannels > 0
        }
    }

    private static func deviceName(for deviceID: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectHasProperty(deviceID, &address) else { return nil }

        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &value) == noErr else {
            return nil
        }
        return value as String
    }

    private static func deviceUID(for deviceID: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectHasProperty(deviceID, &address) else { return nil }

        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &value) == noErr else {
            return nil
        }
        return value as String
    }

    private static func transportType(for deviceID: AudioDeviceID) -> InputDevice.Transport {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard let rawTransportType: UInt32 = readScalarProperty(objectID: deviceID, address: &address, as: UInt32.self) else {
            return .unknown
        }

        switch rawTransportType {
        case kAudioDeviceTransportTypeBuiltIn:
            return .builtIn
        case kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE:
            return .bluetooth
        case kAudioDeviceTransportTypeUSB:
            return .usb
        case kAudioDeviceTransportTypeAggregate:
            return .aggregate
        case kAudioDeviceTransportTypeVirtual:
            return .virtual
        default:
            return .unknown
        }
    }

    private static func readScalarProperty<T: FixedWidthInteger>(
        objectID: AudioObjectID,
        address: inout AudioObjectPropertyAddress,
        as type: T.Type
    ) -> T? {
        guard AudioObjectHasProperty(objectID, &address) else { return nil }
        var value: T = 0
        var size = UInt32(MemoryLayout<T>.size)
        let status = withUnsafeMutablePointer(to: &value) { pointer in
            AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, pointer)
        }
        guard status == noErr else { return nil }
        return value
    }

    private static func readArrayProperty<T>(
        objectID: AudioObjectID,
        address: inout AudioObjectPropertyAddress
    ) -> [T]? {
        guard AudioObjectHasProperty(objectID, &address) else { return nil }
        var propertySize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(objectID, &address, 0, nil, &propertySize) == noErr else {
            return nil
        }
        let count = Int(propertySize) / MemoryLayout<T>.stride
        guard count > 0 else { return [] }

        var values = Array<T>(unsafeUninitializedCapacity: count) { _, initializedCount in
            initializedCount = count
        }
        let status = values.withUnsafeMutableBytes { rawBuffer in
            AudioObjectGetPropertyData(objectID, &address, 0, nil, &propertySize, rawBuffer.baseAddress!)
        }
        guard status == noErr else { return nil }
        return values
    }

    private static func copyAudioBufferList(
        _ source: UnsafeMutablePointer<AudioBufferList>,
        into destination: AVAudioPCMBuffer
    ) {
        let sourceBuffers = UnsafeMutableAudioBufferListPointer(source)
        let destinationBuffers = UnsafeMutableAudioBufferListPointer(destination.mutableAudioBufferList)
        let count = min(sourceBuffers.count, destinationBuffers.count)
        guard count > 0 else { return }

        for index in 0..<count {
            let byteCount = Int(min(sourceBuffers[index].mDataByteSize, destinationBuffers[index].mDataByteSize))
            guard
                byteCount > 0,
                let sourceData = sourceBuffers[index].mData,
                let destinationData = destinationBuffers[index].mData
            else {
                continue
            }
            memcpy(destinationData, sourceData, byteCount)
            destinationBuffers[index].mDataByteSize = UInt32(byteCount)
        }
    }

    private static func formatsMatch(_ lhs: AVAudioFormat, _ rhs: AVAudioFormat) -> Bool {
        lhs.channelCount == rhs.channelCount
            && lhs.commonFormat == rhs.commonFormat
            && lhs.isInterleaved == rhs.isInterleaved
            && abs(lhs.sampleRate - rhs.sampleRate) < 0.5
    }

    private static func describe(device: InputDevice?) -> String {
        guard let device else { return "system-default" }
        return "\(device.name) [\(device.uid)]"
    }

    private static func describe(name: String?, uid: String?) -> String {
        switch (name, uid) {
        case (.some(let name), .some(let uid)):
            return "\(name) [\(uid)]"
        case (.some(let name), .none):
            return name
        case (.none, .some(let uid)):
            return "[\(uid)]"
        case (.none, .none):
            return "system-default"
        }
    }

    private static func describe(outputDevice: MicrophoneRouteCoordinator.OutputDevice?) -> String {
        guard let outputDevice else { return "system-default" }
        return "\(outputDevice.name) [\(outputDevice.uid)]"
    }

    private static func describe(splitRouteArbitration: Bool) -> String {
        splitRouteArbitration ? "engaged" : "skipped"
    }

    private static func describe(format: AVAudioFormat) -> String {
        let commonFormatDescription: String
        switch format.commonFormat {
        case .pcmFormatFloat32:
            commonFormatDescription = "float32"
        case .pcmFormatFloat64:
            commonFormatDescription = "float64"
        case .pcmFormatInt16:
            commonFormatDescription = "int16"
        case .pcmFormatInt32:
            commonFormatDescription = "int32"
        case .otherFormat:
            commonFormatDescription = "other"
        @unknown default:
            commonFormatDescription = "unknown"
        }
        return "\(Int(format.sampleRate))Hz/\(format.channelCount)ch/\(commonFormatDescription)"
    }

    private static func elapsedDescription(since start: DispatchTime) -> String {
        let elapsedNanoseconds = DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds
        let elapsedMilliseconds = Double(elapsedNanoseconds) / 1_000_000
        return String(format: "%.0fms", elapsedMilliseconds)
    }

    private static func backendBanner(
        buildInfoProvider: @Sendable () -> (version: String, build: String)
    ) -> String? {
        backendBannerLock.lock()
        defer { backendBannerLock.unlock() }
        guard !hasLoggedBackendBanner else { return nil }
        hasLoggedBackendBanner = true
        let buildInfo = buildInfoProvider()
        return "[Mic] backend=avcapture version=\(buildInfo.version) build=\(buildInfo.build)"
    }

    static func resetBackendBannerForTesting() {
        backendBannerLock.lock()
        hasLoggedBackendBanner = false
        backendBannerLock.unlock()
    }

    static func convertedFrameCapacity(
        inputFrameLength: AVAudioFrameCount,
        inputSampleRate: Double,
        outputSampleRate: Double
    ) -> AVAudioFrameCount {
        guard inputSampleRate > 0, outputSampleRate > 0 else { return max(inputFrameLength, 1) }
        let exactFrameCount = ceil(Double(inputFrameLength) * outputSampleRate / inputSampleRate)
        return AVAudioFrameCount(max(exactFrameCount, 1.0)) + 1
    }

    static func makeSingleUseInputBlock(buffer: AVAudioPCMBuffer) -> AVAudioConverterInputBlock {
        let state = InputBlockState()
        return { _, outStatus in
            if state.didConsumeBuffer {
                outStatus.pointee = .noDataNow
                return nil
            }
            state.didConsumeBuffer = true
            outStatus.pointee = .haveData
            return buffer
        }
    }
}

private final class InputBlockState: @unchecked Sendable {
    var didConsumeBuffer = false
}

private final class CaptureSession: @unchecked Sendable {
    let outputFormat: AVAudioFormat
    let requestedInputDevice: MicrophoneCaptureService.InputDevice?
    let defaultInputDevice: MicrophoneCaptureService.InputDevice?
    let defaultOutputDevice: MicrophoneRouteCoordinator.OutputDevice?
    let didEngageSplitRouteArbitration: Bool
    let chunkHandler: @Sendable (MicrophoneCaptureService.CaptureChunk) -> Void
    let onStarted: (@Sendable () -> Void)?
    let onFirstInputBuffer: (@Sendable () -> Void)?
    let onFirstChunk: (@Sendable () -> Void)?
    let onError: (@Sendable (Error) -> Void)?
    let logHandler: (@Sendable (String) -> Void)?
    let startedAt: DispatchTime
    var resolvedInputDeviceUID: String?
    var resolvedInputDeviceName: String?
    var controller: (any MicrophoneCaptureControlling)?
    var inputFormat: AVAudioFormat?
    var converter: AVAudioConverter?
    var hasDeliveredFirstInputBuffer = false
    var hasDeliveredFirstChunk = false

    init(
        outputFormat: AVAudioFormat,
        requestedInputDevice: MicrophoneCaptureService.InputDevice?,
        defaultInputDevice: MicrophoneCaptureService.InputDevice?,
        defaultOutputDevice: MicrophoneRouteCoordinator.OutputDevice?,
        didEngageSplitRouteArbitration: Bool,
        resolvedInputDeviceUID: String?,
        resolvedInputDeviceName: String?,
        chunkHandler: @escaping @Sendable (MicrophoneCaptureService.CaptureChunk) -> Void,
        onStarted: (@Sendable () -> Void)?,
        onFirstInputBuffer: (@Sendable () -> Void)?,
        onFirstChunk: (@Sendable () -> Void)?,
        onError: (@Sendable (Error) -> Void)?,
        logHandler: (@Sendable (String) -> Void)?
    ) {
        self.outputFormat = outputFormat
        self.requestedInputDevice = requestedInputDevice
        self.defaultInputDevice = defaultInputDevice
        self.defaultOutputDevice = defaultOutputDevice
        self.didEngageSplitRouteArbitration = didEngageSplitRouteArbitration
        self.resolvedInputDeviceUID = resolvedInputDeviceUID
        self.resolvedInputDeviceName = resolvedInputDeviceName
        self.chunkHandler = chunkHandler
        self.onStarted = onStarted
        self.onFirstInputBuffer = onFirstInputBuffer
        self.onFirstChunk = onFirstChunk
        self.onError = onError
        self.logHandler = logHandler
        self.startedAt = DispatchTime.now()
    }
}

private final class AVFoundationMicrophoneCaptureControllerFactory: MicrophoneCaptureControllerFactory, @unchecked Sendable {
    private let captureDevicesProvider: @Sendable () -> [AVCaptureDevice]

    init(captureDevicesProvider: @escaping @Sendable () -> [AVCaptureDevice] = {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone],
            mediaType: .audio,
            position: .unspecified
        ).devices
    }) {
        self.captureDevicesProvider = captureDevicesProvider
    }

    func makeController(
        preferredInputDeviceUID: String?,
        onPCMBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) throws -> any MicrophoneCaptureControlling {
        let captureDevices = captureDevicesProvider()
        let requestedDevice = preferredInputDeviceUID.flatMap { uid in
            captureDevices.first(where: { $0.uniqueID == uid })
        }
        let fallbackDevice = AVCaptureDevice.default(for: .audio)
            ?? captureDevices.first
        guard let resolvedDevice = requestedDevice ?? fallbackDevice else {
            throw MicrophoneCaptureService.CaptureError.missingCaptureInputDevice
        }

        return try AVFoundationMicrophoneCaptureController(
            requestedInputDeviceUID: preferredInputDeviceUID,
            resolvedDevice: resolvedDevice,
            onPCMBuffer: onPCMBuffer,
            onError: onError
        )
    }
}

private final class AVFoundationMicrophoneCaptureController: NSObject, MicrophoneCaptureControlling, @unchecked Sendable {
    let requestedInputDeviceUID: String?
    let resolvedInputDeviceUID: String?
    let resolvedInputDeviceName: String?

    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "MacAssistant.MicrophoneCaptureService.session")
    private let audioOutput = AVCaptureAudioDataOutput()
    private let outputDelegate: CaptureAudioOutputDelegate
    private let onError: @Sendable (Error) -> Void
    private let controllerStateLock = NSLock()
    private var deviceInput: AVCaptureDeviceInput?
    private var observerTokens: [NSObjectProtocol] = []
    private var isStopped = false

    init(
        requestedInputDeviceUID: String?,
        resolvedDevice: AVCaptureDevice,
        onPCMBuffer: @escaping @Sendable (AVAudioPCMBuffer) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) throws {
        self.requestedInputDeviceUID = requestedInputDeviceUID
        self.resolvedInputDeviceUID = resolvedDevice.uniqueID
        self.resolvedInputDeviceName = resolvedDevice.localizedName
        self.onError = onError
        self.outputDelegate = CaptureAudioOutputDelegate { sampleBuffer in
            guard let pcmBuffer = MicrophoneCaptureService.makePCMBuffer(from: sampleBuffer) else {
                onError(MicrophoneCaptureService.CaptureError.unableToCreatePCMBuffer)
                return
            }
            onPCMBuffer(pcmBuffer)
        }
        super.init()

        do {
            let deviceInput = try AVCaptureDeviceInput(device: resolvedDevice)
            self.deviceInput = deviceInput
            try configureSession(with: deviceInput)
            registerObservers()
        } catch let error as MicrophoneCaptureService.CaptureError {
            throw error
        } catch {
            throw MicrophoneCaptureService.CaptureError.unableToCreateCaptureInput(name: resolvedDevice.localizedName)
        }
    }

    func start() throws {
        try sessionQueue.sync {
            guard !isStopped else {
                throw MicrophoneCaptureService.CaptureError.captureSessionFailedToStart
            }
            session.startRunning()
            guard session.isRunning else {
                throw MicrophoneCaptureService.CaptureError.captureSessionFailedToStart
            }
        }
    }

    func stop() {
        sessionQueue.sync {
            guard markStoppedIfNeeded() else { return }
            audioOutput.setSampleBufferDelegate(nil, queue: nil)
            if session.isRunning {
                session.stopRunning()
            }
            session.beginConfiguration()
            if session.outputs.contains(audioOutput) {
                session.removeOutput(audioOutput)
            }
            if let deviceInput, session.inputs.contains(deviceInput) {
                session.removeInput(deviceInput)
            }
            session.commitConfiguration()
            deviceInput = nil
        }
        removeObservers()
    }

    private func configureSession(with deviceInput: AVCaptureDeviceInput) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        guard session.canAddInput(deviceInput) else {
            throw MicrophoneCaptureService.CaptureError.unableToAddCaptureInput(name: deviceInput.device.localizedName)
        }
        session.addInput(deviceInput)

        audioOutput.audioSettings = [AVFormatIDKey: kAudioFormatLinearPCM]
        audioOutput.setSampleBufferDelegate(outputDelegate, queue: sessionQueue)
        guard session.canAddOutput(audioOutput) else {
            throw MicrophoneCaptureService.CaptureError.unableToAddCaptureOutput
        }
        session.addOutput(audioOutput)
    }

    private func registerObservers() {
        let runtimeErrorToken = NotificationCenter.default.addObserver(
            forName: AVCaptureSession.runtimeErrorNotification,
            object: session,
            queue: nil
        ) { [weak self] notification in
            guard let self, !self.hasStopped else { return }
            if let error = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError {
                self.onError(error)
            } else {
                self.onError(MicrophoneCaptureService.CaptureError.captureSessionFailedToStart)
            }
        }
        let interruptedToken = NotificationCenter.default.addObserver(
            forName: AVCaptureSession.wasInterruptedNotification,
            object: session,
            queue: nil
        ) { [weak self] notification in
            guard let self, !self.hasStopped else { return }
            self.onError(MicrophoneCaptureService.CaptureError.captureSessionInterrupted(reason: "system"))
        }
        observerTokens = [runtimeErrorToken, interruptedToken]
    }

    private func removeObservers() {
        let tokens = observerTokens
        observerTokens.removeAll(keepingCapacity: false)
        for token in tokens {
            NotificationCenter.default.removeObserver(token)
        }
    }

    private var hasStopped: Bool {
        controllerStateLock.lock()
        defer { controllerStateLock.unlock() }
        return isStopped
    }

    private func markStoppedIfNeeded() -> Bool {
        controllerStateLock.lock()
        defer { controllerStateLock.unlock() }
        guard !isStopped else { return false }
        isStopped = true
        return true
    }
}

private final class CaptureAudioOutputDelegate: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    private let sampleBufferHandler: @Sendable (CMSampleBuffer) -> Void

    init(sampleBufferHandler: @escaping @Sendable (CMSampleBuffer) -> Void) {
        self.sampleBufferHandler = sampleBufferHandler
        super.init()
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        sampleBufferHandler(sampleBuffer)
    }
}
