import AppKit
import SwiftUI

private let transcriptBottomAnchor = "transcript-bottom-anchor"
private let transcriptActionButtonSize: CGFloat = 32
private let transcriptActionStripHeight: CGFloat = 36
private let transcriptUserActionHoverWidth: CGFloat = 96

struct ChatView: View {
    @Bindable var model: AppModel
    @State private var expandedImageAttachment: ConversationImageAttachment?
    @FocusState private var isComposerFocused: Bool
    @State private var editingUserMessageID: UUID?
    @State private var editingUserMessageText = ""

    var body: some View {
        VStack(spacing: 0) {
            HeaderBar(model: model, onReset: focusComposerWhenReady)

            Divider()
                .overlay(AppTheme.separator)

            transcript

            Divider()
                .overlay(AppTheme.separator)

            ComposerBar(model: model, isComposerFocused: $isComposerFocused)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(.clear)
        .overlay(alignment: .center) {
            if let attachment = expandedImageAttachment {
                ExpandedMessageImageView(
                    attachment: attachment,
                    onClose: {
                        expandedImageAttachment = nil
                    }
                )
            }
        }
        .onAppear {
            focusComposerWhenReady()
        }
        .onChange(of: model.phase) { _, newPhase in
            if newPhase == .ready {
                focusComposerWhenReady()
            }
        }
    }

    private func focusComposerWhenReady() {
        guard model.phase == .ready else { return }
        DispatchQueue.main.async {
            isComposerFocused = true
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ZStack {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(AppTheme.panelFill)
                    .overlay(
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .strokeBorder(AppTheme.separator, lineWidth: 1)
                    )

                if model.conversation.isEmpty {
                    EmptyTranscriptState()
                        .padding(32)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(model.conversation, id: \.id) { item in
                                itemView(item)
                                    .id(item.id)
                            }

                            Color.clear
                                .frame(height: 1)
                                .id(transcriptBottomAnchor)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 18)
                    }
                    .scrollIndicators(.hidden)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.vertical, 12)
            .onAppear {
                scrollToBottom(proxy)
            }
            .onChange(of: model.conversation) { _, _ in
                scrollToBottom(proxy)
            }
        }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        DispatchQueue.main.async {
            proxy.scrollTo(transcriptBottomAnchor, anchor: .bottom)
        }
    }

    @ViewBuilder
    private func itemView(_ item: ConversationItem) -> some View {
        switch item {
        case .userVoiceDraft(let draft):
            BubbleView(
                text: draft.text.isEmpty ? "Listening…" : draft.text,
                imageAttachment: draft.imageAttachment,
                alignment: .trailing,
                tint: AppTheme.accent.opacity(0.14),
                footer: nil,
                isCancelled: draft.isCancelled,
                rendersMarkdown: false,
                onTapImage: { expandedImageAttachment = $0 }
            )
        case .userMessage(let message):
            UserMessageRow(
                model: model,
                message: message,
                editingUserMessageID: $editingUserMessageID,
                editingUserMessageText: $editingUserMessageText,
                onTapImage: { expandedImageAttachment = $0 }
            )
        case .assistantMessage(let message):
            AssistantMessageRow(
                model: model,
                message: message
            )
        case .toolInvocation(let tool):
            ToolCardView(model: model, tool: tool)
        case .systemStatus(let status):
            Label(status.text, systemImage: status.level == "error" ? "exclamationmark.triangle.fill" : "info.circle.fill")
                .font(.caption)
                .foregroundStyle(status.level == "error" ? AppTheme.destructive : AppTheme.secondaryText)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 2)
        }
    }
}

private struct AssistantMessageRow: View {
    @Bindable var model: AppModel
    let message: AssistantMessage
    @State private var showsCopyConfirmation = false
    @State private var copyFeedbackTask: Task<Void, Never>?

    var body: some View {
        let isSpeaking = model.isSpeakingAssistantMessage(message)

        VStack(alignment: .leading, spacing: 6) {
            BubbleView(
                text: message.text,
                imageAttachment: nil,
                alignment: .leading,
                tint: AppTheme.badgeFill,
                footer: message.isStreaming ? "Generating…" : nil,
                isCancelled: message.isCancelled,
                rendersMarkdown: true,
                onTapImage: { _ in }
            )

            HStack(spacing: 8) {
                TranscriptActionButton(
                    systemImage: showsCopyConfirmation ? "checkmark" : "doc.on.doc",
                    helpText: "Copy response",
                    isDisabled: message.text.isEmpty
                ) {
                    copyResponse()
                }
                TranscriptActionButton(
                    systemImage: isSpeaking ? "stop.fill" : "speaker.wave.2",
                    helpText: isSpeaking ? "Stop reading aloud" : "Read response aloud",
                    isDisabled: !model.canToggleSpeech(for: message)
                ) {
                    model.speakAssistantMessage(message)
                }
                TranscriptActionButton(
                    systemImage: "arrow.clockwise",
                    helpText: "Redo response",
                    isDisabled: model.isTranscriptActionsDisabled || message.isStreaming
                ) {
                    model.redoAssistantMessage(message)
                }
            }
            .padding(.horizontal, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onDisappear {
            copyFeedbackTask?.cancel()
            copyFeedbackTask = nil
        }
    }

    private func copyResponse() {
        model.copyAssistantMessage(message)
        showCopyFeedback()
    }

    private func showCopyFeedback() {
        copyFeedbackTask?.cancel()
        showsCopyConfirmation = true
        copyFeedbackTask = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                showsCopyConfirmation = false
                copyFeedbackTask = nil
            }
        }
    }
}

private struct UserMessageRow: View {
    @Bindable var model: AppModel
    let message: UserMessage
    @Binding var editingUserMessageID: UUID?
    @Binding var editingUserMessageText: String
    let onTapImage: (ConversationImageAttachment) -> Void

    @State private var isActionAreaHovered = false
    @State private var showsCopyConfirmation = false
    @State private var copyFeedbackTask: Task<Void, Never>?
    @FocusState private var isEditorFocused: Bool

    private var isEditing: Bool {
        editingUserMessageID == message.id
    }

    private var canCommitEdit: Bool {
        message.imageAttachment != nil || !editingUserMessageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .trailing, spacing: 0) {
            if isEditing {
                editingBubble
            } else {
                BubbleView(
                    text: message.text,
                    imageAttachment: message.imageAttachment,
                    alignment: .trailing,
                    tint: AppTheme.accent.opacity(0.14),
                    footer: nil,
                    isCancelled: message.isCancelled,
                    rendersMarkdown: false,
                    onTapImage: { onTapImage($0) }
                )
            }

            toolbarStrip
        }
        .onChange(of: isEditing) { _, newValue in
            guard newValue else { return }
            DispatchQueue.main.async {
                isEditorFocused = true
            }
        }
        .onDisappear {
            copyFeedbackTask?.cancel()
            copyFeedbackTask = nil
        }
    }

    private var editingBubble: some View {
        VStack(alignment: .trailing, spacing: message.imageAttachment == nil ? 0 : 10) {
            if let imageAttachment = message.imageAttachment {
                MessageImagePreview(
                    attachment: imageAttachment,
                    open: { onTapImage(imageAttachment) }
                )
            }

            TextField("", text: $editingUserMessageText, axis: .vertical)
                .focused($isEditorFocused)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(1...6)
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .frame(maxWidth: 470, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(AppTheme.accent.opacity(0.14))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(AppTheme.separator, lineWidth: 1)
                )
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
        .padding(.bottom, 6)
    }

    private var toolbarStrip: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 0)

            ZStack(alignment: .trailing) {
                Color.clear
                    .frame(width: transcriptUserActionHoverWidth, height: transcriptActionStripHeight)

                HStack(spacing: 8) {
                    if isEditing {
                        TranscriptActionButton(
                            systemImage: "xmark",
                            helpText: "Cancel edit"
                        ) {
                            cancelEditing()
                        }
                        TranscriptActionButton(
                            systemImage: "checkmark",
                            helpText: "Save edit",
                            isDisabled: model.isTranscriptActionsDisabled || !canCommitEdit
                        ) {
                            commitEditing()
                        }
                    } else {
                        TranscriptActionButton(
                            systemImage: showsCopyConfirmation ? "checkmark" : "doc.on.doc",
                            helpText: "Copy message",
                            isDisabled: message.text.isEmpty
                        ) {
                            copyMessage()
                        }
                        TranscriptActionButton(
                            systemImage: "pencil",
                            helpText: "Edit message",
                            isDisabled: model.isTranscriptActionsDisabled
                        ) {
                            beginEditing()
                        }
                    }
                }
                .opacity(isEditing || isActionAreaHovered ? 1 : 0)
                .allowsHitTesting(isEditing || isActionAreaHovered)
                .animation(.easeOut(duration: 0.14), value: isEditing || isActionAreaHovered)
            }
            .frame(width: transcriptUserActionHoverWidth, height: transcriptActionStripHeight, alignment: .trailing)
            .contentShape(Rectangle())
            .onHover { hovered in
                guard !isEditing else { return }
                isActionAreaHovered = hovered
            }
        }
    }

    private func copyMessage() {
        model.copyUserMessage(message)
        showCopyFeedback()
    }

    private func beginEditing() {
        isActionAreaHovered = false
        editingUserMessageID = message.id
        editingUserMessageText = message.text
    }

    private func cancelEditing() {
        if isEditing {
            editingUserMessageText = message.text
            editingUserMessageID = nil
            isActionAreaHovered = false
        }
    }

    private func commitEditing() {
        guard canCommitEdit else { return }
        editingUserMessageID = nil
        isActionAreaHovered = false
        model.editUserMessage(message, newText: editingUserMessageText)
    }

    private func showCopyFeedback() {
        copyFeedbackTask?.cancel()
        showsCopyConfirmation = true
        copyFeedbackTask = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                showsCopyConfirmation = false
                copyFeedbackTask = nil
            }
        }
    }
}

private struct TranscriptActionButton: View {
    let systemImage: String
    let helpText: String
    var isDisabled = false
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 16, weight: .medium))
                .frame(width: 22, height: 22)
                .foregroundStyle(iconColor)
                .frame(width: transcriptActionButtonSize, height: transcriptActionButtonSize)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(backgroundFill)
                )
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .onHover { hovered in
            isHovered = hovered && !isDisabled
        }
        .animation(.easeOut(duration: 0.12), value: isHovered)
        .help(helpText)
    }

    private var iconColor: Color {
        if isDisabled {
            return AppTheme.tertiaryText
        }
        return isHovered ? .primary : AppTheme.secondaryText
    }

    private var backgroundFill: Color {
        if isDisabled {
            return .clear
        }
        return isHovered ? AppTheme.actionCardHoverFill : .clear
    }
}

private struct HeaderBar: View {
    @Bindable var model: AppModel
    let onReset: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: "waveform.path.ecg")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(AppTheme.accent)
                    .frame(width: 30, height: 30)
                    .background(AppTheme.badgeFill, in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text("MacAssistant")
                        .font(.headline)
                    Text(model.statusText)
                        .font(.caption)
                        .foregroundStyle(AppTheme.secondaryText)
                        .lineLimit(1)
                }

                Spacer(minLength: 0)

                HStack(spacing: 8) {
                    Button {
                        model.resetConversation()
                        onReset()
                    } label: {
                        Label("Reset", systemImage: "arrow.counterclockwise")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                    .disabled(!model.canResetConversation)

                    Button(role: .destructive) {
                        model.stopActiveTurn()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)

                    Button {
                        model.showingSettings = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .frame(width: 16, height: 16)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                }
            }

            HStack(spacing: 8) {
                StatusBadge(
                    title: "Agent",
                    icon: "brain.head.profile",
                    state: model.underlyingModels[RuntimeModelID.agentModel.rawValue]?.warmState ?? .cold
                )
                StatusBadge(
                    title: "Voice",
                    icon: "speaker.wave.2",
                    state: model.underlyingModels[RuntimeModelID.ttsModel.rawValue]?.warmState ?? .cold
                )
                StatusBadge(
                    title: "Mic",
                    icon: "mic",
                    state: model.underlyingModels[RuntimeModelID.sttModel.rawValue]?.warmState ?? .cold
                )

                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 2)
        .padding(.bottom, 12)
    }
}

private struct StatusBadge: View {
    let title: String
    let icon: String
    let state: WarmState

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)

            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))

            Text(title)
                .lineLimit(1)
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(AppTheme.secondaryText)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(AppTheme.badgeFill, in: Capsule())
        .overlay(
            Capsule()
                .strokeBorder(AppTheme.separator, lineWidth: 1)
        )
        .fixedSize(horizontal: true, vertical: false)
    }

    private var color: Color {
        switch state {
        case .cold:
            return AppTheme.warning
        case .warming:
            return AppTheme.accent
        case .warm:
            return AppTheme.success
        case .error:
            return AppTheme.destructive
        }
    }
}

private struct EmptyTranscriptState: View {
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "sparkles")
                .font(.system(size: 22, weight: .medium))
                .foregroundStyle(AppTheme.accent)

            Text("Start a conversation")
                .font(.title3.weight(.semibold))

            Text("Talk to your Mac or type a request. Tool activity appears inline only when the assistant needs to act.")
                .font(.subheadline)
                .foregroundStyle(AppTheme.secondaryText)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
    }
}

private struct ComposerBar: View {
    @Bindable var model: AppModel
    @FocusState.Binding var isComposerFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let attachment = model.pendingComposerImage {
                ComposerAttachmentPreview(
                    attachment: attachment,
                    isDisabled: model.isRecording || model.isComposerLocked,
                    remove: model.removePendingComposerImage
                )
            }

            HStack(alignment: .bottom, spacing: 10) {
                Button {
                    model.promptForImageAttachment()
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.plain)
                .foregroundStyle((model.isRecording || model.isComposerLocked) ? AppTheme.tertiaryText : Color.primary)
                .background(
                    Circle()
                        .fill(AppTheme.badgeFill)
                )
                .disabled(model.isRecording || model.isComposerLocked)

                TextField("Message MacAssistant", text: $model.composerText, axis: .vertical)
                    .focused($isComposerFocused)
                    .textFieldStyle(.plain)
                    .font(.body)
                    .lineLimit(1...4)
                    .padding(.horizontal, 13)
                    .padding(.vertical, 11)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(AppTheme.inputFill)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(AppTheme.separator, lineWidth: 1)
                    )
                    .disabled(model.hasLiveVoiceDraft)
                    .onSubmit(model.sendCurrentInput)

                Button {
                    if showsStopOutputButton {
                        model.stopActiveTurn()
                    } else {
                        model.sendCurrentInput()
                    }
                } label: {
                    Image(systemName: sendButtonIconName)
                        .font(.system(size: 14, weight: .semibold))
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.plain)
                .foregroundStyle(sendButtonForegroundStyle)
                .background(
                    Circle()
                        .fill(sendButtonBackgroundStyle)
                )
                .disabled(!isSendButtonEnabled)

                Button {
                    if model.isRecording {
                        model.discardCurrentInput()
                    } else {
                        model.toggleRecording()
                    }
                } label: {
                    Image(systemName: model.isRecording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.plain)
                .foregroundStyle(model.isRecording ? Color.white : Color.primary)
                .background(
                    Circle()
                        .fill(model.isRecording ? AppTheme.destructive : AppTheme.badgeFill)
                )
                .disabled(model.isFinalizingVoiceRecording || (!model.isRecording && (model.isComposerLocked || !model.isMicReady)))
                .overlay(alignment: .topTrailing) {
                    if !model.hasLiveVoiceDraft && !model.isMicReady && model.phase == .ready {
                        ProgressView()
                            .scaleEffect(0.55)
                            .offset(x: 1, y: -1)
                    }
                }
            }

            HStack {
                Text(helperText)
                    .font(.caption)
                    .foregroundStyle(AppTheme.tertiaryText)

                Spacer()

                if model.isStopping {
                    Text("Stopping…")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(AppTheme.secondaryText)
                }
            }
        }
        .padding(.top, 12)
    }

    private var canSend: Bool {
        if model.isRecording {
            return true
        }
        if model.hasLiveVoiceDraft {
            return false
        }
        if model.isComposerLocked {
            return false
        }
        return model.pendingComposerImage != nil
            || !model.composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var showsStopOutputButton: Bool {
        model.canStopCurrentOutput && !model.hasLiveVoiceDraft
    }

    private var sendButtonIconName: String {
        showsStopOutputButton ? "stop.fill" : "arrow.up"
    }

    private var isSendButtonEnabled: Bool {
        showsStopOutputButton || canSend
    }

    private var sendButtonForegroundStyle: Color {
        isSendButtonEnabled ? .white : AppTheme.tertiaryText
    }

    private var sendButtonBackgroundStyle: Color {
        isSendButtonEnabled ? AppTheme.accent : AppTheme.badgeFill
    }

    private var helperText: String {
        if model.isWaitingForMicrophoneAudio {
            return "Waiting for microphone audio. Press send to finish, or stop to discard the draft."
        }
        if model.isRecording {
            return "Listening for speech. Press send to use it, or stop to discard it."
        }
        if model.isFinalizingVoiceRecording {
            return "Finalizing speech. The live transcript will submit as soon as the last chunk is processed."
        }
        if model.isReplySpeechActive {
            return "Reading the response aloud. Press stop to silence it."
        }
        if model.isComposerLocked {
            return "MacAssistant is working. You can keep typing, or press stop to cancel the current response."
        }
        return "Press Return to send, or use the mic for a spoken request."
    }
}

private struct BubbleView: View {
    let text: String
    let imageAttachment: ConversationImageAttachment?
    let alignment: HorizontalAlignment
    let tint: Color
    let footer: String?
    let isCancelled: Bool
    let rendersMarkdown: Bool
    let onTapImage: (ConversationImageAttachment) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if alignment == .trailing {
                Spacer(minLength: 0)
            }

            VStack(alignment: alignment, spacing: resolvedFooter == nil ? 0 : 4) {
                VStack(alignment: alignment, spacing: imageAttachment == nil ? 0 : 10) {
                    if let imageAttachment {
                        MessageImagePreview(
                            attachment: imageAttachment,
                            open: { onTapImage(imageAttachment) }
                        )
                    }

                    ViewThatFits(in: .horizontal) {
                        BubbleText(
                            text: text,
                            rendersMarkdown: rendersMarkdown,
                            textAlignment: .leading
                        )
                            .fixedSize(horizontal: true, vertical: true)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 11)
                            .background(
                                RoundedRectangle(cornerRadius: 18, style: .continuous)
                                    .fill(tint)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 18, style: .continuous)
                                    .strokeBorder(AppTheme.separator, lineWidth: 1)
                            )
                            .opacity(isCancelled ? 0.52 : 1)

                        BubbleText(
                            text: text,
                            rendersMarkdown: rendersMarkdown,
                            textAlignment: .leading
                        )
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 11)
                            .frame(maxWidth: 470, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 18, style: .continuous)
                                    .fill(tint)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 18, style: .continuous)
                                    .strokeBorder(AppTheme.separator, lineWidth: 1)
                            )
                            .opacity(isCancelled ? 0.52 : 1)
                    }
                }

                if let resolvedFooter {
                    Text(resolvedFooter)
                        .font(.caption)
                        .foregroundStyle(AppTheme.tertiaryText)
                        .padding(.horizontal, 4)
                }
            }

            if alignment == .leading {
                Spacer(minLength: 0)
            }
        }
    }

    private var resolvedFooter: String? {
        switch (footer, isCancelled) {
        case let (footer?, true):
            return "\(footer) · Cancelled"
        case (nil, true):
            return "Cancelled"
        case let (footer?, false):
            return footer
        case (nil, false):
            return nil
        }
    }
}

private struct ComposerAttachmentPreview: View {
    let attachment: ComposerImageAttachment
    let isDisabled: Bool
    let remove: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            AttachmentThumbnail(url: attachment.fileURL, size: 62, cornerRadius: 12)

            VStack(alignment: .leading, spacing: 4) {
                Text("Image attached")
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
                Text("Will be sent with your next message.")
                    .font(.caption)
                    .foregroundStyle(AppTheme.secondaryText)
            }

            Spacer(minLength: 0)

            Button {
                remove()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.plain)
            .foregroundStyle(AppTheme.secondaryText)
            .background(AppTheme.badgeFill, in: Circle())
            .disabled(isDisabled)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(AppTheme.panelFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(AppTheme.separator, lineWidth: 1)
        )
    }
}

private struct MessageImagePreview: View {
    let attachment: ConversationImageAttachment
    let open: () -> Void

    var body: some View {
        Button(action: open) {
            AttachmentThumbnail(url: attachment.fileURL, size: 84, cornerRadius: 16)
        }
        .buttonStyle(.plain)
        .help("Click to enlarge")
    }
}

private struct ExpandedMessageImageView: View {
    let attachment: ConversationImageAttachment
    let onClose: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.opacity(0.55)
                .ignoresSafeArea()
                .onTapGesture {
                    onClose()
                }

            Group {
                if let image = NSImage(contentsOf: attachment.fileURL) {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                } else {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(AppTheme.badgeFill)
                        .overlay {
                            Image(systemName: "photo")
                                .font(.system(size: 28, weight: .medium))
                                .foregroundStyle(AppTheme.secondaryText)
                        }
                }
            }
            .frame(maxWidth: 920, maxHeight: 720)
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
            .padding(32)
            .onTapGesture {}

            Button {
                onClose()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .semibold))
                    .frame(width: 30, height: 30)
            }
            .buttonStyle(.plain)
            .background(AppTheme.badgeFill, in: Circle())
            .overlay(
                Circle()
                    .strokeBorder(AppTheme.separator, lineWidth: 1)
            )
            .padding(20)
        }
        .frame(minWidth: 720, minHeight: 520)
    }
}

private struct AttachmentThumbnail: View {
    let url: URL
    let size: CGFloat
    let cornerRadius: CGFloat
    var contentMode: ContentMode = .fill

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(AppTheme.badgeFill)

            if let image = NSImage(contentsOf: url) {
                Group {
                    if contentMode == .fill {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                            .padding(18)
                    }
                }
            } else {
                Image(systemName: "photo")
                    .font(.system(size: 20, weight: .medium))
                    .foregroundStyle(AppTheme.secondaryText)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(AppTheme.separator, lineWidth: 1)
        )
    }
}

private struct BubbleText: View {
    let text: String
    let rendersMarkdown: Bool
    let textAlignment: TextAlignment

    var body: some View {
        Group {
            if rendersMarkdown, let markdown = parsedMarkdown {
                Text(markdown)
            } else {
                Text(text)
            }
        }
        .font(.body)
        .lineSpacing(2)
        .textSelection(.enabled)
        .multilineTextAlignment(textAlignment)
        .tint(AppTheme.accent)
    }

    private var parsedMarkdown: AttributedString? {
        try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .full,
                failurePolicy: .returnPartiallyParsedIfPossible
            )
        )
    }
}

private struct ToolCardView: View {
    @Bindable var model: AppModel
    let tool: ToolInvocation

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button {
                guard canToggleExpansion else { return }
                model.toggleExpansion(for: tool)
            } label: {
                HStack(spacing: 10) {
                    Group {
                        if showsPreparingIndicator {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: tool.isExpanded ? "chevron.down" : "chevron.right")
                                .font(.system(size: 11, weight: .semibold))
                        }
                    }
                    .foregroundStyle(showsPreparingIndicator ? AppTheme.accent : AppTheme.secondaryText)
                    .frame(width: 14, height: 14)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(tool.summary)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(canToggleExpansion ? Color.primary : AppTheme.secondaryText)
                        Text(subtitle)
                            .font(.caption)
                            .foregroundStyle(subtitleColor)
                    }

                    Spacer(minLength: 0)

                    Text(statusLabel)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(executionColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(executionColor.opacity(0.14), in: Capsule())
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!canToggleExpansion)

            if tool.isExpanded {
                VStack(alignment: .leading, spacing: 10) {
                    InfoBlock(title: "Arguments", content: pretty(tool.arguments))

                    if !tool.output.isEmpty {
                        InfoBlock(title: "Result", content: tool.output)
                    }

                    if tool.approvalState == .pending {
                        HStack(spacing: 10) {
                            Button("Allow") {
                                model.approveTool(tool)
                            }
                            .buttonStyle(.borderedProminent)

                            Button("Not Now") {
                                model.denyTool(tool)
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(AppTheme.panelFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(AppTheme.separator, lineWidth: 1)
        )
        .animation(.easeOut(duration: 0.18), value: tool.isExpanded)
    }

    private var executionColor: Color {
        switch tool.executionState {
        case .proposed:
            return tool.approvalState == .pending ? AppTheme.warning : AppTheme.accent
        case .running:
            return AppTheme.accent
        case .finished:
            return AppTheme.success
        case .failed, .cancelled:
            return AppTheme.destructive
        }
    }

    private var canToggleExpansion: Bool {
        tool.isExpanded || tool.approvalState == .pending || !tool.arguments.isEmpty || !tool.output.isEmpty || tool.executionState != .proposed
    }

    private var showsPreparingIndicator: Bool {
        !tool.isExpanded && !canToggleExpansion
    }

    private var subtitle: String {
        if showsPreparingIndicator {
            return "Preparing details…"
        }
        if tool.approvalState == .pending {
            return tool.isExpanded ? "Review this action before it runs." : "Click to review before it runs."
        }
        if tool.executionState == .running {
            return tool.isExpanded ? "The assistant is running this action now." : "Running. Click to inspect progress."
        }
        if !tool.output.isEmpty {
            return tool.isExpanded ? "Result shown below." : "Click to inspect the result."
        }
        return tool.name
    }

    private var subtitleColor: Color {
        if showsPreparingIndicator {
            return AppTheme.accent
        }
        if tool.approvalState == .pending {
            return AppTheme.secondaryText
        }
        return AppTheme.secondaryText
    }

    private var statusLabel: String {
        if showsPreparingIndicator {
            return "Preparing"
        }
        if tool.approvalState == .pending {
            return tool.isExpanded ? "Needs Approval" : "Review"
        }
        switch tool.executionState {
        case .proposed:
            return "Proposed"
        case .running:
            return "Running"
        case .finished:
            return "Finished"
        case .failed:
            return "Failed"
        case .cancelled:
            return "Cancelled"
        }
    }

    private func pretty(_ arguments: [String: JSONValue]) -> String {
        guard let data = try? JSONEncoder().encode(arguments),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }
}

private struct InfoBlock: View {
    let title: String
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(AppTheme.secondaryText)

            Text(content)
                .font(.system(size: 12, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(AppTheme.inputFill.opacity(0.72))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(AppTheme.separator, lineWidth: 1)
                )
        }
    }
}
