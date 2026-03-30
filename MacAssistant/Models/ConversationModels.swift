import Foundation

enum TurnSource: String, Codable, Sendable {
    case voice
    case typed
}

struct ConversationImageAttachment: Identifiable, Equatable, Sendable {
    let id: UUID
    let fileURL: URL
    let displayName: String
}

struct ComposerImageAttachment: Identifiable, Equatable, Sendable {
    let id: UUID
    let fileURL: URL
    let displayName: String
}

enum SafetyClass: String, Codable, Sendable, CaseIterable {
    case readOnly
    case mutating
}

enum ApprovalState: String, Codable, Sendable {
    case notRequired
    case pending
    case approved
    case denied
}

enum ExecutionState: String, Codable, Sendable {
    case proposed
    case running
    case finished
    case failed
    case cancelled
}

struct UserVoiceDraft: Identifiable, Equatable, Sendable {
    let id: UUID
    let turnID: String
    var text: String
    var imageAttachment: ConversationImageAttachment?
    var source: TurnSource
    var isCancelled: Bool
}

struct UserMessage: Identifiable, Equatable, Sendable {
    let id: UUID
    let turnID: String
    var text: String
    var imageAttachment: ConversationImageAttachment?
    var source: TurnSource
    var isCancelled: Bool
}

struct AssistantMessage: Identifiable, Equatable, Sendable {
    let id: UUID
    let turnID: String
    let segmentID: String
    var text: String
    var isStreaming: Bool
    var isCancelled: Bool
    var isFinalSegment: Bool
    var source: TurnSource
}

struct ToolInvocation: Identifiable, Equatable, Sendable {
    let id: UUID
    let turnID: String
    let toolCallID: String
    var name: String
    var arguments: [String: JSONValue]
    var safetyClass: SafetyClass
    var approvalState: ApprovalState
    var executionState: ExecutionState
    var summary: String
    var output: String
    var isExpanded: Bool
}

struct SystemStatusMessage: Identifiable, Equatable, Sendable {
    let id: UUID
    let turnID: String?
    var text: String
    var level: String
}

enum ConversationItem: Identifiable, Equatable, Sendable {
    case userVoiceDraft(UserVoiceDraft)
    case userMessage(UserMessage)
    case assistantMessage(AssistantMessage)
    case toolInvocation(ToolInvocation)
    case systemStatus(SystemStatusMessage)

    var id: UUID {
        switch self {
        case .userVoiceDraft(let item):
            return item.id
        case .userMessage(let item):
            return item.id
        case .assistantMessage(let item):
            return item.id
        case .toolInvocation(let item):
            return item.id
        case .systemStatus(let item):
            return item.id
        }
    }

    var turnID: String? {
        switch self {
        case .userVoiceDraft(let item):
            return item.turnID
        case .userMessage(let item):
            return item.turnID
        case .assistantMessage(let item):
            return item.turnID
        case .toolInvocation(let item):
            return item.turnID
        case .systemStatus(let item):
            return item.turnID
        }
    }
}
