@preconcurrency import AVFoundation
import AudioToolbox
import CoreAudio
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

final class MicrophoneCaptureService: MicrophoneCaptureServicing, @unchecked Sendable {
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
        case missingInputAudioUnit
        case unableToBuildOutputFormat
        case unableToCreateConverter
        case unableToCreateConvertedBuffer
        case unableToSelectInputDevice(name: String, status: OSStatus)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Microphone access is required to record voice input."
            case .missingInputFormat:
                return "No microphone input format is available."
            case .missingInputAudioUnit:
                return "Unable to access the system microphone audio unit."
            case .unableToBuildOutputFormat:
                return "Unable to create the voice capture output format."
            case .unableToCreateConverter:
                return "Unable to convert microphone audio into the speech input format."
            case .unableToCreateConvertedBuffer:
                return "Unable to allocate the converted microphone audio buffer."
            case .unableToSelectInputDevice(let name, let status):
                return "Unable to select microphone input device “\(name)” (OSStatus \(status))."
            }
        }
    }

    static let targetSampleRate = 16_000.0
    static let targetChannels: AVAudioChannelCount = 1
    static let tapBufferSize: AVAudioFrameCount = 1_024

    var onInputDevicesChanged: (@Sendable () -> Void)?

    private let stateLock = NSLock()
    private let audioDeviceNotificationQueue = DispatchQueue(label: "MacAssistant.MicrophoneCaptureService.devices")
    private var activeSession: CaptureSession?

    private lazy var devicesChangedListener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
        self?.onInputDevicesChanged?()
    }

    private lazy var defaultInputChangedListener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
        self?.onInputDevicesChanged?()
    }

    init() {
        installAudioDeviceListeners()
    }

    deinit {
        removeAudioDeviceListeners()
    }

    func authorizationStatus() -> AuthorizationStatus {
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
    }

    func requestAccess() async -> Bool {
        let status = authorizationStatus()
        if status == .authorized {
            return true
        }
        if status == .denied {
            return false
        }

        return await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func availableInputDevices() -> [InputDevice] {
        Self.enumerateInputDevices()
    }

    func defaultInputDevice() -> InputDevice? {
        guard let defaultDeviceID = Self.defaultInputDeviceID() else { return nil }
        return availableInputDevices().first(where: { $0.deviceID == defaultDeviceID })
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

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let availableDevices = availableInputDevices()
        let requestedInputDevice = preferredInputDeviceUID.flatMap { uid in
            availableDevices.first(where: { $0.uid == uid })
        }
        if let preferredInputDeviceUID, requestedInputDevice == nil {
            logHandler?("[Mic] Requested input device uid=\(preferredInputDeviceUID) is unavailable. Falling back to the active system route.")
        }

        if let requestedInputDevice {
            try Self.selectInputDevice(requestedInputDevice, on: input)
        }

        let preStartInputFormat = input.outputFormat(forBus: 0)
        guard preStartInputFormat.channelCount > 0 else {
            throw CaptureError.missingInputFormat
        }
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Self.targetSampleRate,
            channels: Self.targetChannels,
            interleaved: true
        ) else {
            throw CaptureError.unableToBuildOutputFormat
        }

        let resolvedInputDevice = Self.currentInputDevice(for: input, availableDevices: availableDevices)
            ?? requestedInputDevice
            ?? defaultInputDevice()
        let session = CaptureSession(
            engine: engine,
            outputFormat: outputFormat,
            requestedInputDevice: requestedInputDevice,
            activeInputDevice: resolvedInputDevice,
            chunkHandler: chunkHandler,
            onStarted: onStarted,
            onFirstInputBuffer: onFirstInputBuffer,
            onFirstChunk: onFirstChunk,
            onError: onError,
            logHandler: logHandler
        )
        session.configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            self?.handleConfigurationChange(for: session)
        }
        setSession(session)

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: Self.tapBufferSize, format: nil) { [weak self] buffer, _ in
            self?.handleBuffer(buffer, session: session)
        }

        do {
            engine.prepare()
            try engine.start()
        } catch {
            clearSessionIfNeeded(session)
            teardown(session: session)
            session.logHandler?(
                "[Mic] Failed to start capture requested=\(Self.describe(device: session.requestedInputDevice)) " +
                "resolved=\(Self.describe(device: session.activeInputDevice)) " +
                "input=\(Self.describe(format: preStartInputFormat)): \(error.localizedDescription)"
            )
            throw error
        }

        let startedInputFormat = input.outputFormat(forBus: 0)
        session.activeInputDevice = Self.currentInputDevice(for: input, availableDevices: availableDevices)
            ?? session.activeInputDevice
        session.logHandler?(
            "[Mic] Engine started device=\(Self.describe(device: session.activeInputDevice)) " +
            "requested=\(Self.describe(device: session.requestedInputDevice)) " +
            "input=\(Self.describe(format: startedInputFormat)) output=\(Self.describe(format: outputFormat))"
        )
        if !Self.formatsMatch(preStartInputFormat, startedInputFormat) {
            session.logHandler?(
                "[Mic] Input format changed during start preflight=\(Self.describe(format: preStartInputFormat)) " +
                "postStart=\(Self.describe(format: startedInputFormat))"
            )
        }
        session.onStarted?()
    }

    func stop() {
        guard let session = takeCurrentSession() else { return }
        teardown(session: session)
        session.logHandler?(
            "[Mic] Stopped capture after \(Self.elapsedDescription(since: session.startedAt)) " +
            "(device=\(Self.describe(device: session.activeInputDevice)), rawInput=\(session.hasDeliveredFirstInputBuffer), convertedChunk=\(session.hasDeliveredFirstChunk))"
        )
    }

    private func handleConfigurationChange(for session: CaptureSession) {
        guard isCurrentSession(session) else { return }
        let activeDevices = availableInputDevices()
        session.activeInputDevice = Self.currentInputDevice(for: session.engine.inputNode, availableDevices: activeDevices)
            ?? session.activeInputDevice
        let currentFormat = session.engine.inputNode.outputFormat(forBus: 0)
        session.logHandler?(
            "[Mic] Configuration changed device=\(Self.describe(device: session.activeInputDevice)) " +
            "input=\(Self.describe(format: currentFormat))"
        )
    }

    private func handleBuffer(_ buffer: AVAudioPCMBuffer, session: CaptureSession) {
        guard isCurrentSession(session) else { return }

        session.activeInputDevice = Self.currentInputDevice(for: session.engine.inputNode, availableDevices: availableInputDevices())
            ?? session.activeInputDevice
        guard let converter = ensureConverter(for: buffer.format, session: session) else { return }

        if markFirstInputBufferIfNeeded(for: session) {
            session.logHandler?(
                "[Mic] First input buffer after \(Self.elapsedDescription(since: session.startedAt)) " +
                "device=\(Self.describe(device: session.activeInputDevice)) format=\(Self.describe(format: buffer.format))"
            )
            session.onFirstInputBuffer?()
        }

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
                "device=\(Self.describe(device: session.activeInputDevice)) input=\(Self.describe(format: buffer.format)) " +
                "output=\(Self.describe(format: session.outputFormat))"
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
        if let configurationObserver = session.configurationObserver {
            NotificationCenter.default.removeObserver(configurationObserver)
            session.configurationObserver = nil
        }
        session.engine.inputNode.removeTap(onBus: 0)
        session.engine.stop()
        session.engine.reset()
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

    private static func defaultInputDeviceID() -> AudioDeviceID? {
        var defaultInputAddress = defaultInputPropertyAddress()
        return readScalarProperty(
            objectID: AudioObjectID(kAudioObjectSystemObject),
            address: &defaultInputAddress,
            as: AudioDeviceID.self
        )
    }

    private static func currentInputDevice(
        for inputNode: AVAudioInputNode,
        availableDevices: [InputDevice]
    ) -> InputDevice? {
        guard let audioUnit = inputNode.audioUnit else { return nil }

        var deviceID = AudioDeviceID(0)
        var propertySize = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioUnitGetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            &propertySize
        )
        guard status == noErr else { return nil }

        return availableDevices.first(where: { $0.deviceID == deviceID })
            ?? enumerateInputDevices().first(where: { $0.deviceID == deviceID })
    }

    private static func selectInputDevice(_ device: InputDevice, on inputNode: AVAudioInputNode) throws {
        guard let audioUnit = inputNode.audioUnit else {
            throw CaptureError.missingInputAudioUnit
        }

        var mutableDeviceID = device.deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &mutableDeviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        guard status == noErr else {
            throw CaptureError.unableToSelectInputDevice(name: device.name, status: status)
        }
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

        var values = Array<T>(unsafeUninitializedCapacity: count) { buffer, initializedCount in
            initializedCount = count
        }
        let status = values.withUnsafeMutableBytes { rawBuffer in
            AudioObjectGetPropertyData(objectID, &address, 0, nil, &propertySize, rawBuffer.baseAddress!)
        }
        guard status == noErr else { return nil }
        return values
    }

    private static func formatsMatch(_ lhs: AVAudioFormat, _ rhs: AVAudioFormat) -> Bool {
        lhs.channelCount == rhs.channelCount
            && lhs.commonFormat == rhs.commonFormat
            && abs(lhs.sampleRate - rhs.sampleRate) < 0.5
    }

    private static func describe(device: InputDevice?) -> String {
        guard let device else { return "system-default" }
        return "\(device.name) [\(device.uid)]"
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
    let engine: AVAudioEngine
    let outputFormat: AVAudioFormat
    let requestedInputDevice: MicrophoneCaptureService.InputDevice?
    let chunkHandler: @Sendable (MicrophoneCaptureService.CaptureChunk) -> Void
    let onStarted: (@Sendable () -> Void)?
    let onFirstInputBuffer: (@Sendable () -> Void)?
    let onFirstChunk: (@Sendable () -> Void)?
    let onError: (@Sendable (Error) -> Void)?
    let logHandler: (@Sendable (String) -> Void)?
    let startedAt: DispatchTime
    var activeInputDevice: MicrophoneCaptureService.InputDevice?
    var inputFormat: AVAudioFormat?
    var converter: AVAudioConverter?
    var configurationObserver: NSObjectProtocol?
    var hasDeliveredFirstInputBuffer = false
    var hasDeliveredFirstChunk = false

    init(
        engine: AVAudioEngine,
        outputFormat: AVAudioFormat,
        requestedInputDevice: MicrophoneCaptureService.InputDevice?,
        activeInputDevice: MicrophoneCaptureService.InputDevice?,
        chunkHandler: @escaping @Sendable (MicrophoneCaptureService.CaptureChunk) -> Void,
        onStarted: (@Sendable () -> Void)?,
        onFirstInputBuffer: (@Sendable () -> Void)?,
        onFirstChunk: (@Sendable () -> Void)?,
        onError: (@Sendable (Error) -> Void)?,
        logHandler: (@Sendable (String) -> Void)?
    ) {
        self.engine = engine
        self.outputFormat = outputFormat
        self.requestedInputDevice = requestedInputDevice
        self.activeInputDevice = activeInputDevice
        self.chunkHandler = chunkHandler
        self.onStarted = onStarted
        self.onFirstInputBuffer = onFirstInputBuffer
        self.onFirstChunk = onFirstChunk
        self.onError = onError
        self.logHandler = logHandler
        self.startedAt = DispatchTime.now()
    }
}
