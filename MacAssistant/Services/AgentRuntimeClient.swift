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
    private var process: Process?
    private var stdinHandle: FileHandle?
    private var stdoutHandle: FileHandle?
    private var stderrHandle: FileHandle?
    private var stdoutBuffer = Data()
    private var stderrBuffer = Data()
    private var isStopping = false

    var isRunning: Bool {
        process?.isRunning == true
    }

    init(appSupportURL: URL) {
        self.appSupportURL = appSupportURL
    }

    func start() throws {
        guard process == nil else { return }
        isStopping = false

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
        self.stdoutBuffer = Data()
        self.stderrBuffer = Data()

        configureReadabilityHandlers()
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
        stdoutBuffer = Data()
        stderrBuffer = Data()
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

    private func configureReadabilityHandlers() {
        stdoutHandle?.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self else { return }
            Task { @MainActor in
                self.consumeOutputChunk(data, stream: .stdout)
            }
        }

        stderrHandle?.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self else { return }
            Task { @MainActor in
                self.consumeOutputChunk(data, stream: .stderr)
            }
        }
    }

    private func consumeOutputChunk(_ data: Data, stream: StreamKind) {
        guard !data.isEmpty else {
            switch stream {
            case .stdout:
                stdoutHandle?.readabilityHandler = nil
            case .stderr:
                stderrHandle?.readabilityHandler = nil
            }
            return
        }

        switch stream {
        case .stdout:
            stdoutBuffer.append(data)
            drainBuffer(&stdoutBuffer, stream: .stdout)
        case .stderr:
            stderrBuffer.append(data)
            drainBuffer(&stderrBuffer, stream: .stderr)
        }
    }

    private func drainBuffer(_ buffer: inout Data, stream: StreamKind) {
        while let newlineIndex = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[..<newlineIndex]
            buffer.removeSubrange(...newlineIndex)
            guard let line = String(data: lineData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                !line.isEmpty else { continue }

            if Self.shouldSuppressRuntimeLog(line) {
                continue
            }

            switch stream {
            case .stdout:
                writeLaunchLog("stdout: \(line)")
                if line.hasPrefix("{") {
                    do {
                        let event = try JSONDecoder().decode(RuntimeEventEnvelope.self, from: Data(line.utf8))
                        onEvent?(event)
                    } catch {
                        onError?("Failed to decode runtime event: \(line)")
                    }
                } else {
                    onLog?("stdout: \(line)")
                }
            case .stderr:
                writeLaunchLog("stderr: \(line)")
                onLog?("stderr: \(line)")
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
        let formatter = ISO8601DateFormatter()
        let line = "[\(formatter.string(from: Date()))] \(message)\n"
        let logURL = appSupportURL.appendingPathComponent("logs/runtime-launch.log", isDirectory: false)
        try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let data = line.data(using: .utf8) {
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

private enum StreamKind {
    case stdout
    case stderr
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
