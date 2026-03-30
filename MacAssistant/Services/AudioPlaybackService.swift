import AVFoundation
import Foundation

@MainActor
protocol AudioPlaybackServicing: AnyObject {
    var onPlaybackStarted: (() -> Void)? { get set }
    var onPlaybackFinished: (() -> Void)? { get set }

    func enqueuePCMData(_ data: Data, sampleRate: Int, channels: Int, encoding: String)
    func finishStream()
    func stop()
}

@MainActor
final class AudioPlaybackService: AudioPlaybackServicing {
    var onPlaybackStarted: (() -> Void)?
    var onPlaybackFinished: (() -> Void)?

    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let startBufferDuration: TimeInterval
    private var activeFormat: AVAudioFormat?
    private var queuedFrameCount: AVAudioFramePosition = 0
    private var pendingBufferCount = 0
    private var hasStartedPlayback = false
    private var streamFinished = false

    init(startBufferDuration: TimeInterval = 0.25) {
        self.startBufferDuration = startBufferDuration
        engine.attach(playerNode)
    }

    func enqueuePCMData(_ data: Data, sampleRate: Int, channels: Int, encoding: String) {
        guard encoding.lowercased() == "pcm_s16le" else {
            NSLog("Unsupported playback encoding: \(encoding)")
            return
        }
        guard let format = configureFormat(sampleRate: sampleRate, channels: channels) else {
            return
        }
        guard let buffer = makePCMBuffer(from: data, format: format, channels: channels) else {
            return
        }

        let frameCount = AVAudioFramePosition(buffer.frameLength)
        pendingBufferCount += 1
        queuedFrameCount += frameCount
        streamFinished = false

        playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                self?.handleBufferPlayback(frameCount: frameCount)
            }
        }
        startPlaybackIfReady(force: false)
    }

    func finishStream() {
        streamFinished = true
        if pendingBufferCount == 0 {
            stopEngine(resetFormat: false)
            streamFinished = false
            onPlaybackFinished?()
            return
        }
        startPlaybackIfReady(force: true)
    }

    func stop() {
        stopEngine(resetFormat: true)
        streamFinished = false
    }

    private func configureFormat(sampleRate: Int, channels: Int) -> AVAudioFormat? {
        guard sampleRate > 0, channels > 0 else { return nil }
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channels)
        ) else {
            return nil
        }

        let needsReconnect =
            activeFormat?.sampleRate != format.sampleRate
            || activeFormat?.channelCount != format.channelCount
        if needsReconnect || activeFormat == nil {
            stopEngine(resetFormat: false)
            engine.disconnectNodeOutput(playerNode)
            engine.connect(playerNode, to: engine.mainMixerNode, format: format)
            engine.prepare()
            activeFormat = format
        }

        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                NSLog("Failed to start playback engine: \(error.localizedDescription)")
                return nil
            }
        }

        return activeFormat
    }

    private func makePCMBuffer(from data: Data, format: AVAudioFormat, channels: Int) -> AVAudioPCMBuffer? {
        let bytesPerSample = MemoryLayout<Int16>.stride
        let sampleCount = data.count / bytesPerSample
        guard sampleCount > 0, sampleCount % channels == 0 else { return nil }
        let frameCount = sampleCount / channels
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(frameCount)
        ) else {
            return nil
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)

        let samples: [Int16] = data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Int16.self))
        }

        guard let channelData = buffer.floatChannelData else { return nil }
        for frame in 0..<frameCount {
            let baseIndex = frame * channels
            for channel in 0..<channels {
                channelData[channel][frame] = Float(samples[baseIndex + channel]) / 32768.0
            }
        }
        return buffer
    }

    private func startPlaybackIfReady(force: Bool) {
        guard !hasStartedPlayback, let format = activeFormat else { return }
        let queuedDuration = Double(queuedFrameCount) / format.sampleRate
        guard force || queuedDuration >= startBufferDuration else { return }
        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                NSLog("Failed to restart playback engine: \(error.localizedDescription)")
                return
            }
        }
        playerNode.play()
        hasStartedPlayback = true
        onPlaybackStarted?()
    }

    private func handleBufferPlayback(frameCount: AVAudioFramePosition) {
        queuedFrameCount = max(0, queuedFrameCount - frameCount)
        pendingBufferCount = max(0, pendingBufferCount - 1)
        guard pendingBufferCount == 0 else { return }

        let shouldNotify = hasStartedPlayback || streamFinished
        stopEngine(resetFormat: false)
        let wasFinalBuffer = streamFinished
        streamFinished = false

        if shouldNotify {
            onPlaybackFinished?()
        }

        if wasFinalBuffer {
            activeFormat = nil
        }
    }

    private func stopEngine(resetFormat: Bool) {
        if playerNode.isPlaying {
            playerNode.stop()
        }
        if engine.isRunning {
            engine.stop()
        }
        queuedFrameCount = 0
        pendingBufferCount = 0
        hasStartedPlayback = false
        if resetFormat {
            activeFormat = nil
        }
    }
}
