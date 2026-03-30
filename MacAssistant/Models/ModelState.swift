import Foundation

enum ModelRole: String, Codable, Sendable {
    case voicePack
    case agent
}

enum ModelInstallState: String, Codable, Sendable, Equatable {
    case missing
    case downloading
    case installed
    case failed
}

enum WarmState: String, Codable, Sendable, Equatable {
    case cold
    case warming
    case warm
    case error
}

struct UnderlyingModelState: Equatable, Sendable {
    let id: String
    var installState: ModelInstallState
    var warmState: WarmState
    var lastError: String?
}

struct InstallableDownloadProgress: Equatable, Sendable {
    let installableID: String
    let modelID: String?
    var bytesDownloaded: Int64
    var bytesTotal: Int64
    var speedBytesPerSecond: Double?
    var etaSeconds: Double?
    var message: String?

    var fractionCompleted: Double? {
        guard bytesTotal > 0 else { return nil }
        let fraction = Double(bytesDownloaded) / Double(bytesTotal)
        return min(max(fraction, 0), 1)
    }
}

struct ModelInstallable: Identifiable, Equatable, Sendable {
    let id: String
    let role: ModelRole
    let title: String
    let summary: String
    let downloadSize: String
    let modelIDs: [String]
    var installState: ModelInstallState
    var warmState: WarmState
    var deleteAllowed: Bool
    var lastError: String?

    static let defaults: [ModelInstallable] = [
        ModelInstallable(
            id: "voice_pack",
            role: .voicePack,
            title: "Voice Pack",
            summary: "Bundles local speech input and speech output models for a full voice loop.",
            downloadSize: "~5.66 GB",
            modelIDs: ["stt_model", "tts_model"],
            installState: .missing,
            warmState: .cold,
            deleteAllowed: false,
            lastError: nil
        ),
        ModelInstallable(
            id: "agent_model",
            role: .agent,
            title: "Agent Model",
            summary: "Local multimodal Qwen 3.5 model for chat, reasoning, tool orchestration, and image understanding.",
            downloadSize: "~3.03 GB",
            modelIDs: ["agent_model"],
            installState: .missing,
            warmState: .cold,
            deleteAllowed: false,
            lastError: nil
        )
    ]
}
