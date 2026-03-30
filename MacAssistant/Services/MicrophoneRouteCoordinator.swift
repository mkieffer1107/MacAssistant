@preconcurrency import AVFoundation
import CoreAudio
import Foundation

protocol MicrophoneRouteCoordinating: AnyObject, Sendable {
    func prepareForRecording(inputDevice: MicrophoneCaptureService.InputDevice?) async throws -> MicrophoneRouteCoordinator.PreflightResult
    func cancelPendingPreparation()
    func leaveRecordingRoute()
    func routeStateSnapshot() -> MicrophoneRouteCoordinator.RouteState
}

final class MicrophoneRouteCoordinator: NSObject, MicrophoneRouteCoordinating, @unchecked Sendable {
    struct OutputDevice: Equatable, Sendable {
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

        var isBluetooth: Bool { transport == .bluetooth }
    }

    struct PreflightResult: Equatable, Sendable {
        let defaultOutputDevice: OutputDevice?
        let didEngageArbitration: Bool
    }

    struct RouteState: Equatable, Sendable {
        let defaultOutputDevice: OutputDevice?
        let isArbitrationActive: Bool
    }

    enum PreflightError: LocalizedError {
        case arbitrationFailed(message: String)

        var errorDescription: String? {
            switch self {
            case .arbitrationFailed(let message):
                return message
            }
        }
    }

    static let shared = MicrophoneRouteCoordinator()

    private let stateLock = NSLock()
    private let defaultOutputDeviceProvider: @Sendable () -> OutputDevice?
    private let beginArbitrationHandler: @Sendable (@escaping @Sendable (Bool, Error?) -> Void) -> Void
    private let leaveArbitrationHandler: @Sendable () -> Void
    private var pendingRequestID: UUID?
    private var pendingContinuation: CheckedContinuation<Void, Error>?
    private var cancelledRequestIDs = Set<UUID>()
    private var arbitrationActive = false

    init(
        defaultOutputDeviceProvider: @escaping @Sendable () -> OutputDevice? = {
            MicrophoneRouteCoordinator.resolveDefaultOutputDevice()
        },
        beginArbitrationHandler: @escaping @Sendable (@escaping @Sendable (Bool, Error?) -> Void) -> Void = { completion in
            AVAudioRoutingArbiter.shared.begin(category: .playAndRecordVoice, completionHandler: completion)
        },
        leaveArbitrationHandler: @escaping @Sendable () -> Void = {
            AVAudioRoutingArbiter.shared.leave()
        }
    ) {
        self.defaultOutputDeviceProvider = defaultOutputDeviceProvider
        self.beginArbitrationHandler = beginArbitrationHandler
        self.leaveArbitrationHandler = leaveArbitrationHandler
        super.init()
    }

    func prepareForRecording(inputDevice: MicrophoneCaptureService.InputDevice?) async throws -> PreflightResult {
        let defaultOutputDevice = defaultOutputDeviceProvider()
        let shouldEngageArbitration = shouldEngageSplitRouteArbitration(
            inputDevice: inputDevice,
            defaultOutputDevice: defaultOutputDevice
        )
        guard shouldEngageArbitration else {
            return PreflightResult(defaultOutputDevice: defaultOutputDevice, didEngageArbitration: false)
        }

        try await beginArbitration()
        if Task.isCancelled {
            leaveRecordingRoute()
            throw CancellationError()
        }

        stateLock.withLock {
            arbitrationActive = true
        }
        return PreflightResult(defaultOutputDevice: defaultOutputDevice, didEngageArbitration: true)
    }

    func cancelPendingPreparation() {
        cancelPendingRequest()
    }

    func leaveRecordingRoute() {
        let shouldLeave = stateLock.withLock { () -> Bool in
            let shouldLeave = arbitrationActive
            arbitrationActive = false
            return shouldLeave
        }

        if shouldLeave {
            leaveArbitrationHandler()
        }
    }

    func routeStateSnapshot() -> RouteState {
        let isArbitrationActive = stateLock.withLock { arbitrationActive }
        return RouteState(
            defaultOutputDevice: defaultOutputDeviceProvider(),
            isArbitrationActive: isArbitrationActive
        )
    }

    private func beginArbitration() async throws {
        let requestID = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                registerPendingRequest(id: requestID, continuation: continuation)
                beginArbitrationHandler { [weak self] _, error in
                    self?.finishPendingRequest(id: requestID, error: error)
                }
            }
        } onCancel: { [weak self] in
            self?.cancelPendingRequest(id: requestID)
        }
    }

    private func shouldEngageSplitRouteArbitration(
        inputDevice: MicrophoneCaptureService.InputDevice?,
        defaultOutputDevice: OutputDevice?
    ) -> Bool {
        guard let inputDevice, let defaultOutputDevice else { return false }
        guard !inputDevice.isBluetooth else { return false }
        return defaultOutputDevice.isBluetooth
    }

    private func registerPendingRequest(id: UUID, continuation: CheckedContinuation<Void, Error>) {
        stateLock.withLock {
            pendingRequestID = id
            pendingContinuation = continuation
        }
    }

    private func cancelPendingRequest() {
        let requestID = stateLock.withLock { pendingRequestID }
        if let requestID {
            cancelPendingRequest(id: requestID)
        }
    }

    private func cancelPendingRequest(id: UUID) {
        let continuation = stateLock.withLock { () -> CheckedContinuation<Void, Error>? in
            if pendingRequestID == id {
                let continuation = pendingContinuation
                pendingRequestID = nil
                pendingContinuation = nil
                cancelledRequestIDs.insert(id)
                return continuation
            }
            return nil
        }

        continuation?.resume(throwing: CancellationError())
        leaveArbitrationHandler()
    }

    private func finishPendingRequest(id: UUID, error: Error?) {
        let (continuation, wasCancelled) = stateLock.withLock { () -> (CheckedContinuation<Void, Error>?, Bool) in
            if pendingRequestID == id {
                let continuation = pendingContinuation
                pendingRequestID = nil
                pendingContinuation = nil
                return (continuation, false)
            }
            return (nil, cancelledRequestIDs.remove(id) != nil)
        }

        if wasCancelled {
            if error == nil {
                leaveArbitrationHandler()
            }
            return
        }

        if let error {
            continuation?.resume(throwing: PreflightError.arbitrationFailed(message: error.localizedDescription))
        } else {
            continuation?.resume(returning: ())
        }
    }

    private static func defaultOutputPropertyAddress() -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    private static func resolveDefaultOutputDevice() -> OutputDevice? {
        var defaultOutputAddress = defaultOutputPropertyAddress()
        guard
            let defaultDeviceID = readScalarProperty(
                objectID: AudioObjectID(kAudioObjectSystemObject),
                address: &defaultOutputAddress,
                as: AudioDeviceID.self
            )
        else {
            return nil
        }

        let uid = deviceUID(for: defaultDeviceID) ?? "device-\(defaultDeviceID)"
        let name = deviceName(for: defaultDeviceID) ?? uid
        return OutputDevice(
            deviceID: defaultDeviceID,
            uid: uid,
            name: name,
            transport: transportType(for: defaultDeviceID)
        )
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
        let status = withUnsafeMutablePointer(to: &value) { pointer in
            AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, pointer)
        }
        guard status == noErr else {
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
        let status = withUnsafeMutablePointer(to: &value) { pointer in
            AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, pointer)
        }
        guard status == noErr else {
            return nil
        }
        return value as String
    }

    private static func transportType(for deviceID: AudioDeviceID) -> OutputDevice.Transport {
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
}

final class NoopMicrophoneRouteCoordinator: MicrophoneRouteCoordinating, @unchecked Sendable {
    static let shared = NoopMicrophoneRouteCoordinator()

    func prepareForRecording(inputDevice: MicrophoneCaptureService.InputDevice?) async throws -> MicrophoneRouteCoordinator.PreflightResult {
        .init(defaultOutputDevice: nil, didEngageArbitration: false)
    }

    func cancelPendingPreparation() {}

    func leaveRecordingRoute() {}

    func routeStateSnapshot() -> MicrophoneRouteCoordinator.RouteState {
        .init(defaultOutputDevice: nil, isArbitrationActive: false)
    }
}
