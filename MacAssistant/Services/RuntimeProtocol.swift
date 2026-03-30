import Foundation

struct RuntimeAttachment: Codable, Equatable, Sendable {
    let type: String
    let filePath: String
    let displayName: String
}

struct RuntimeHistoryMessage: Codable, Equatable, Sendable {
    let role: String
    let text: String
    let attachments: [RuntimeAttachment]?
}

struct RuntimeCommandEnvelope: Codable, Equatable, Sendable {
    let type: String
    let turnID: String?
    let modelID: String?
    let installableID: String?
    let text: String?
    let speakReply: Bool?
    let chunkBase64: String?
    let sampleRate: Int?
    let channels: Int?
    let audioEncoding: String?
    let toolCallID: String?
    let attachments: [RuntimeAttachment]?
    let arguments: [String: JSONValue]?
    let history: [RuntimeHistoryMessage]?

    init(
        type: String,
        turnID: String? = nil,
        modelID: String? = nil,
        installableID: String? = nil,
        text: String? = nil,
        speakReply: Bool? = nil,
        chunkBase64: String? = nil,
        sampleRate: Int? = nil,
        channels: Int? = nil,
        audioEncoding: String? = nil,
        toolCallID: String? = nil,
        attachments: [RuntimeAttachment]? = nil,
        arguments: [String: JSONValue]? = nil,
        history: [RuntimeHistoryMessage]? = nil
    ) {
        self.type = type
        self.turnID = turnID
        self.modelID = modelID
        self.installableID = installableID
        self.text = text
        self.speakReply = speakReply
        self.chunkBase64 = chunkBase64
        self.sampleRate = sampleRate
        self.channels = channels
        self.audioEncoding = audioEncoding
        self.toolCallID = toolCallID
        self.attachments = attachments
        self.arguments = arguments
        self.history = history
    }
}

struct RuntimeEventEnvelope: Codable, Sendable {
    let type: String
    let turnID: String?
    let modelID: String?
    let installableID: String?
    let installState: String?
    let warmState: String?
    let message: String?
    let stage: String?
    let text: String?
    let progress: Double?
    let bytesDownloaded: Int64?
    let bytesTotal: Int64?
    let etaSeconds: Double?
    let speedBytesPerSecond: Double?
    let chunkBase64: String?
    let sampleRate: Int?
    let channels: Int?
    let audioEncoding: String?
    let toolCallID: String?
    let toolName: String?
    let toolSummary: String?
    let toolArguments: [String: JSONValue]?
    let safetyClass: String?
    let approvalState: String?
    let executionState: String?
    let output: String?
    let status: String?
    let isFinal: Bool?

    init(
        type: String,
        turnID: String? = nil,
        modelID: String? = nil,
        installableID: String? = nil,
        installState: String? = nil,
        warmState: String? = nil,
        message: String? = nil,
        stage: String? = nil,
        text: String? = nil,
        progress: Double? = nil,
        bytesDownloaded: Int64? = nil,
        bytesTotal: Int64? = nil,
        etaSeconds: Double? = nil,
        speedBytesPerSecond: Double? = nil,
        chunkBase64: String? = nil,
        sampleRate: Int? = nil,
        channels: Int? = nil,
        audioEncoding: String? = nil,
        toolCallID: String? = nil,
        toolName: String? = nil,
        toolSummary: String? = nil,
        toolArguments: [String: JSONValue]? = nil,
        safetyClass: String? = nil,
        approvalState: String? = nil,
        executionState: String? = nil,
        output: String? = nil,
        status: String? = nil,
        isFinal: Bool? = nil
    ) {
        self.type = type
        self.turnID = turnID
        self.modelID = modelID
        self.installableID = installableID
        self.installState = installState
        self.warmState = warmState
        self.message = message
        self.stage = stage
        self.text = text
        self.progress = progress
        self.bytesDownloaded = bytesDownloaded
        self.bytesTotal = bytesTotal
        self.etaSeconds = etaSeconds
        self.speedBytesPerSecond = speedBytesPerSecond
        self.chunkBase64 = chunkBase64
        self.sampleRate = sampleRate
        self.channels = channels
        self.audioEncoding = audioEncoding
        self.toolCallID = toolCallID
        self.toolName = toolName
        self.toolSummary = toolSummary
        self.toolArguments = toolArguments
        self.safetyClass = safetyClass
        self.approvalState = approvalState
        self.executionState = executionState
        self.output = output
        self.status = status
        self.isFinal = isFinal
    }
}

enum RuntimeModelID: String, CaseIterable, Sendable {
    case agentModel = "agent_model"
    case ttsModel = "tts_model"
    case sttModel = "stt_model"
}

extension RuntimeModelID {
    var title: String {
        switch self {
        case .agentModel:
            return "Agent Model"
        case .ttsModel:
            return "Speech Output"
        case .sttModel:
            return "Speech Input"
        }
    }

    var summary: String {
        switch self {
        case .agentModel:
            return "Primary chat, reasoning, and tool orchestration model."
        case .ttsModel:
            return "Text-to-speech model used for spoken replies."
        case .sttModel:
            return "Speech-to-text model used for live transcription."
        }
    }

    var systemImage: String {
        switch self {
        case .agentModel:
            return "brain.head.profile"
        case .ttsModel:
            return "speaker.wave.2"
        case .sttModel:
            return "mic"
        }
    }
}
