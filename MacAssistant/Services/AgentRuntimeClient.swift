@preconcurrency import Foundation
import Darwin

@MainActor
protocol RuntimeClienting: AnyObject {
    var onEvent: (@Sendable @MainActor (RuntimeEventEnvelope) -> Void)? { get set }
    var onError: (@Sendable @MainActor (String) -> Void)? { get set }
    var onLog: (@Sendable @MainActor (String) -> Void)? { get set }
    var onTermination: (@Sendable @MainActor (String) -> Void)? { get set }
    var lastSentCommand: RuntimeCommandEnvelope? { get }
    var isRunning: Bool { get }

    func start() throws
    func stop(gracefully: Bool)
    func send(_ command: RuntimeCommandEnvelope) throws
}

@MainActor
final class AgentRuntimeClient: RuntimeClienting {
    var onEvent: (@Sendable @MainActor (RuntimeEventEnvelope) -> Void)?
    var onError: (@Sendable @MainActor (String) -> Void)?
    var onLog: (@Sendable @MainActor (String) -> Void)?
    var onTermination: (@Sendable @MainActor (String) -> Void)?
    private(set) var lastSentCommand: RuntimeCommandEnvelope?

    private let appSupportURL: URL
    private let launchLogWriter: RuntimeLaunchLogWriter
    private var process: Process?
    private var stdinHandle: FileHandle?
    private var stdoutHandle: FileHandle?
    private var stderrHandle: FileHandle?
    private lazy var outputPump = RuntimeOutputPump(logWriter: launchLogWriter)
    private var runtimeOutputSessionID = UUID()
    private var isStopping = false

    var isRunning: Bool {
        process?.isRunning == true
    }

    init(appSupportURL: URL) {
        self.appSupportURL = appSupportURL
        self.launchLogWriter = RuntimeLaunchLogWriter(
            logURL: appSupportURL.appendingPathComponent("logs/runtime-launch.log", isDirectory: false)
        )
    }

    func start() throws {
        guard process == nil else { return }
        isStopping = false
        runtimeOutputSessionID = UUID()
        outputPump.reset()

        let scriptName = "AgentRuntimeHost.sh"
        guard let scriptURL = Bundle.main.url(forResource: "AgentRuntimeHost", withExtension: "sh") else {
            throw RuntimeLaunchError.missingScript(scriptName)
        }

        writeLaunchLog("Starting runtime host from \(scriptURL.path)")

        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()

        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = [scriptURL.path]
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        process.environment = mergedEnvironment()
        let terminationHandler = onTermination
        process.terminationHandler = { process in
            Task { @MainActor in
                self.writeLaunchLog("Runtime terminated with code \(process.terminationStatus)")
                if !self.isStopping {
                    terminationHandler?("Runtime exited with code \(process.terminationStatus).")
                }
            }
        }

        try process.run()

        self.process = process
        self.stdinHandle = stdin.fileHandleForWriting
        self.stdoutHandle = stdout.fileHandleForReading
        self.stderrHandle = stderr.fileHandleForReading

        configureReadabilityHandlers(sessionID: runtimeOutputSessionID)
        writeLaunchLog("Runtime process started")
    }

    func stop(gracefully: Bool = true) {
        isStopping = true
        stdoutHandle?.readabilityHandler = nil
        stderrHandle?.readabilityHandler = nil
        if gracefully {
            sendShutdownCommandIfPossible()
            waitForExit(timeout: 1.5)
        }
        stdinHandle?.closeFile()
        if let process, process.isRunning {
            process.terminate()
            waitForExit(timeout: 0.5)
        }
        if let process, process.isRunning {
            process.interrupt()
            waitForExit(timeout: 0.25)
        }
        if let process, process.isRunning {
            kill(process.processIdentifier, SIGKILL)
        }
        stdinHandle = nil
        stdoutHandle = nil
        stderrHandle = nil
        outputPump.reset()
        self.process = nil
    }

    func send(_ command: RuntimeCommandEnvelope) throws {
        lastSentCommand = command
        guard let stdinHandle else {
            throw RuntimeLaunchError.runtimeUnavailable
        }
        let encoded = try JSONEncoder().encode(command)
        var payload = encoded
        payload.append(0x0A)
        try stdinHandle.write(contentsOf: payload)
        if let json = String(data: encoded, encoding: .utf8) {
            writeLaunchLog("Sent command: \(json)")
        }
    }

    private func configureReadabilityHandlers(sessionID: UUID) {
        let outputPump = self.outputPump

        stdoutHandle?.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self else { return }
            guard !data.isEmpty else {
                Task { @MainActor in
                    guard self.runtimeOutputSessionID == sessionID else { return }
                    self.stdoutHandle?.readabilityHandler = nil
                }
                return
            }
            outputPump.enqueue(data, stream: .stdout) { [weak self] outputs in
                Task { @MainActor in
                    guard let self, self.runtimeOutputSessionID == sessionID else { return }
                    self.handleProcessedOutputs(outputs)
                }
            }
        }

        stderrHandle?.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self else { return }
            guard !data.isEmpty else {
                Task { @MainActor in
                    guard self.runtimeOutputSessionID == sessionID else { return }
                    self.stderrHandle?.readabilityHandler = nil
                }
                return
            }
            outputPump.enqueue(data, stream: .stderr) { [weak self] outputs in
                Task { @MainActor in
                    guard let self, self.runtimeOutputSessionID == sessionID else { return }
                    self.handleProcessedOutputs(outputs)
                }
            }
        }
    }

    private func handleProcessedOutputs(_ outputs: [RuntimeProcessedOutput]) {
        for output in outputs {
            switch output {
            case .event(let event):
                onEvent?(event)
            case .log(let line):
                onLog?(line)
            case .error(let message):
                onError?(message)
            }
        }
    }

    private func mergedEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let homeDirectory = FileManager.default.homeDirectoryForCurrentUser.path
        let preferredPathEntries = [
            "\(homeDirectory)/.local/bin",
            "\(homeDirectory)/.bun/bin",
            "/opt/homebrew/bin",
            "/opt/homebrew/sbin",
            "/usr/local/bin",
            "/usr/local/sbin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ]
        let existingPathEntries = (environment["PATH"] ?? "")
            .split(separator: ":")
            .map(String.init)
        let mergedPathEntries = preferredPathEntries + existingPathEntries.filter { !preferredPathEntries.contains($0) }

        environment["MAC_ASSISTANT_APP_SUPPORT"] = appSupportURL.path
        environment["MAC_ASSISTANT_BUNDLE_RESOURCES"] = Bundle.main.resourceURL?.path ?? ""
        environment["PATH"] = mergedPathEntries.joined(separator: ":")
        environment["BUN_INSTALL"] = "\(homeDirectory)/.bun"
        return environment
    }

    private func writeLaunchLog(_ message: String) {
        launchLogWriter.write(message)
    }

    private func sendShutdownCommandIfPossible() {
        guard let stdinHandle else { return }
        let command = RuntimeCommandEnvelope(
            type: "shutdown",
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
        )
        lastSentCommand = command
        guard let payload = try? JSONEncoder().encode(command) else { return }
        var line = payload
        line.append(0x0A)
        try? stdinHandle.write(contentsOf: line)
        if let json = String(data: payload, encoding: .utf8) {
            writeLaunchLog("Sent command: \(json)")
        }
    }

    private func waitForExit(timeout: TimeInterval) {
        guard let process, process.isRunning else { return }
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.05))
        }
    }

    nonisolated static func shouldSuppressRuntimeLog(_ line: String) -> Bool {
        let suppressedPrefixes = [
            "Loaded Mistral tokenizer from ",
            "Loaded local Tekken tokenizer from ",
            "Loaded HF tokenizer from ",
            "Resolving dependencies",
            "Resolved, downloaded and extracted [",
            "Saved lockfile",
        ]
        if suppressedPrefixes.contains(where: { line.hasPrefix($0) }) {
            return true
        }

        let suppressedFragments = [
            "[macos_automator_server] [INFO] [Server Startup] Current working directory",
            "[macos_automator_server] [INFO] Starting macos_automator MCP Server...",
            "[macos_automator_server] [WARN] CRITICAL: Ensure macOS Automation & Accessibility permissions are correctly configured",
            "[KnowledgeBaseManager] [INFO] KB_PARSING is lazy (or not set). Knowledge base will load on first use.",
        ]
        return suppressedFragments.contains(where: line.contains)
    }
}

private enum StreamKind: Sendable {
    case stdout
    case stderr
}

private enum RuntimeProcessedOutput: Sendable {
    case event(RuntimeEventEnvelope)
    case log(String)
    case error(String)
}

private final class RuntimeLaunchLogWriter: @unchecked Sendable {
    private let logURL: URL
    private let queue = DispatchQueue(label: "MacAssistant.RuntimeLaunchLogWriter", qos: .utility)

    init(logURL: URL) {
        self.logURL = logURL
    }

    func write(_ message: String) {
        let logURL = self.logURL
        queue.async {
            let formatter = ISO8601DateFormatter()
            let line = "[\(formatter.string(from: Date()))] \(message)\n"
            guard let data = line.data(using: .utf8) else { return }
            try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: logURL.path) {
                if let handle = try? FileHandle(forWritingTo: logURL) {
                    _ = try? handle.seekToEnd()
                    try? handle.write(contentsOf: data)
                    try? handle.close()
                }
            } else {
                try? data.write(to: logURL)
            }
        }
    }
}

private final class RuntimeOutputPump: @unchecked Sendable {
    private let queue = DispatchQueue(label: "MacAssistant.RuntimeOutputPump", qos: .userInitiated)
    private let logWriter: RuntimeLaunchLogWriter
    private var stdoutBuffer = Data()
    private var stderrBuffer = Data()

    init(logWriter: RuntimeLaunchLogWriter) {
        self.logWriter = logWriter
    }

    func enqueue(
        _ data: Data,
        stream: StreamKind,
        deliver: @escaping @Sendable ([RuntimeProcessedOutput]) -> Void
    ) {
        queue.async { [weak self] in
            guard let self else { return }
            let outputs = self.consume(data, stream: stream)
            guard !outputs.isEmpty else { return }
            deliver(outputs)
        }
    }

    func reset() {
        queue.async { [weak self] in
            self?.stdoutBuffer.removeAll(keepingCapacity: false)
            self?.stderrBuffer.removeAll(keepingCapacity: false)
        }
    }

    private func consume(_ data: Data, stream: StreamKind) -> [RuntimeProcessedOutput] {
        switch stream {
        case .stdout:
            stdoutBuffer.append(data)
            return drain(buffer: &stdoutBuffer, stream: .stdout)
        case .stderr:
            stderrBuffer.append(data)
            return drain(buffer: &stderrBuffer, stream: .stderr)
        }
    }

    private func drain(buffer: inout Data, stream: StreamKind) -> [RuntimeProcessedOutput] {
        var outputs: [RuntimeProcessedOutput] = []

        while let newlineIndex = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[..<newlineIndex]
            buffer.removeSubrange(...newlineIndex)
            guard let line = String(data: lineData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                !line.isEmpty else { continue }

            if AgentRuntimeClient.shouldSuppressRuntimeLog(line) {
                continue
            }

            switch stream {
            case .stdout:
                logWriter.write("stdout: \(line)")
                if line.hasPrefix("{") {
                    do {
                        let event = try JSONDecoder().decode(RuntimeEventEnvelope.self, from: Data(line.utf8))
                        outputs.append(.event(event))
                    } catch {
                        outputs.append(.error("Failed to decode runtime event: \(line)"))
                    }
                } else {
                    outputs.append(.log("stdout: \(line)"))
                }
            case .stderr:
                logWriter.write("stderr: \(line)")
                outputs.append(.log("stderr: \(line)"))
            }
        }

        return outputs
    }
}

enum RuntimeLaunchError: LocalizedError, Equatable {
    case missingScript(String)
    case runtimeUnavailable

    var errorDescription: String? {
        switch self {
        case .missingScript(let file):
            return "Missing runtime script resource: \(file)"
        case .runtimeUnavailable:
            return "The runtime process is not available."
        }
    }
}
