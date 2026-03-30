import SwiftUI

struct SettingsSectionCard<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            content
        }
        .padding(16)
        .sectionSurface(cornerRadius: 20, material: .thinMaterial, shadowOpacity: 0.03)
    }
}

struct SettingsSectionHeader<Accessory: View>: View {
    let systemImage: String
    let title: String
    let subtitle: String
    private let accessory: Accessory

    init(
        systemImage: String,
        title: String,
        subtitle: String,
        @ViewBuilder accessory: () -> Accessory
    ) {
        self.systemImage = systemImage
        self.title = title
        self.subtitle = subtitle
        self.accessory = accessory()
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(AppTheme.secondaryText)
                .frame(width: 22, height: 22)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.title3.weight(.semibold))
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
            }

            Spacer(minLength: 0)

            accessory
        }
    }
}

extension SettingsSectionHeader where Accessory == EmptyView {
    init(systemImage: String, title: String, subtitle: String) {
        self.init(systemImage: systemImage, title: title, subtitle: subtitle) {
            EmptyView()
        }
    }
}

struct SettingsRow<Control: View>: View {
    let title: String
    let subtitle: String
    let controlWidth: CGFloat?
    private let control: Control

    init(
        title: String,
        subtitle: String,
        controlWidth: CGFloat? = nil,
        @ViewBuilder control: () -> Control
    ) {
        self.title = title
        self.subtitle = subtitle
        self.controlWidth = controlWidth
        self.control = control()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.headline)
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Group {
                if let controlWidth {
                    control
                        .frame(width: controlWidth, alignment: .trailing)
                } else {
                    control
                        .fixedSize()
                }
            }
        }
        .padding(.vertical, 8)
    }
}

struct SettingsRowDivider: View {
    var body: some View {
        Divider()
            .overlay(AppTheme.separator)
    }
}

struct ModelFileRow: View {
    let title: String
    let summary: String
    let systemImage: String
    let installState: ModelInstallState
    let warmState: WarmState
    let lastError: String?
    let progress: InstallableDownloadProgress?
    let actionTitle: String?
    let action: ModelFileAction?
    let onAction: (ModelFileAction) -> Void

    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        formatter.zeroPadsFractionDigits = false
        return formatter
    }()

    private static let durationFormatter: DateComponentsFormatter = {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute]
        formatter.unitsStyle = .abbreviated
        formatter.maximumUnitCount = 2
        return formatter
    }()

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            iconTile

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.title3.weight(.semibold))

                Text(summary)
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    SettingsStatusBadge(
                        title: installState.rawValue.capitalized,
                        color: installStateColor(for: installState)
                    )

                    if let warmBadgeTitle {
                        SettingsStatusBadge(
                            title: warmBadgeTitle,
                            color: warmStateColor(for: warmState)
                        )
                    }
                }

                if let progress, installState == .downloading {
                    VStack(alignment: .leading, spacing: 6) {
                        ProgressView(value: progress.fractionCompleted)
                            .progressViewStyle(.linear)
                            .tint(AppTheme.accent)

                        HStack(spacing: 10) {
                            Text(downloadProgressSummary(progress))
                            Text("•")
                            Text(downloadProgressTiming(progress))
                        }
                        .font(.caption)
                        .foregroundStyle(AppTheme.secondaryText)
                    }
                }

                if let lastError {
                    Label(lastError, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(AppTheme.destructive)
                }
            }

            Spacer(minLength: 0)

            if let actionTitle, let action {
                Button(actionTitle) {
                    onAction(action)
                }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                    .padding(.top, 2)
            }
        }
        .padding(.vertical, 10)
    }

    private var iconTile: some View {
        Image(systemName: systemImage)
            .font(.system(size: 18, weight: .semibold))
            .foregroundStyle(AppTheme.accent)
            .frame(width: 42, height: 42)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(AppTheme.actionCardFill)
            )
    }

    private var warmBadgeTitle: String? {
        switch warmState {
        case .cold:
            return nil
        case .warming:
            return "Warming"
        case .warm:
            return "Warm"
        case .error:
            return "Error"
        }
    }

    private func installStateColor(for state: ModelInstallState) -> Color {
        switch state {
        case .missing:
            return AppTheme.warning
        case .downloading:
            return AppTheme.accent
        case .installed:
            return AppTheme.success
        case .failed:
            return AppTheme.destructive
        }
    }

    private func warmStateColor(for state: WarmState) -> Color {
        switch state {
        case .cold:
            return AppTheme.secondaryText
        case .warming:
            return AppTheme.accent
        case .warm:
            return AppTheme.success
        case .error:
            return AppTheme.destructive
        }
    }

    private func downloadProgressSummary(_ progress: InstallableDownloadProgress) -> String {
        guard progress.bytesTotal > 0 else {
            return "Preparing download…"
        }

        return "\(formattedByteCount(progress.bytesDownloaded)) of \(formattedByteCount(progress.bytesTotal))"
    }

    private func downloadProgressTiming(_ progress: InstallableDownloadProgress) -> String {
        if let etaSeconds = progress.etaSeconds,
           etaSeconds.isFinite,
           etaSeconds > 1,
           let formattedETA = Self.durationFormatter.string(from: etaSeconds) {
            return "About \(formattedETA) left"
        }

        if let speedBytesPerSecond = progress.speedBytesPerSecond,
           speedBytesPerSecond.isFinite,
           speedBytesPerSecond > 0 {
            return "\(formattedByteCount(Int64(speedBytesPerSecond)))/s"
        }

        return progress.bytesTotal > 0 ? "Calculating time…" : "Estimating size…"
    }

    private func formattedByteCount(_ bytes: Int64) -> String {
        Self.byteFormatter.string(fromByteCount: bytes)
    }
}

enum ModelFileAction: Equatable, Sendable {
    case installInstallable(String)
    case deleteModel(RuntimeModelID)
}

struct ModelFileEntry: Identifiable, Equatable, Sendable {
    let id: String
    let title: String
    let summary: String
    let systemImage: String
    let installState: ModelInstallState
    let warmState: WarmState
    let lastError: String?
    let progress: InstallableDownloadProgress?
    let actionTitle: String?
    let action: ModelFileAction?
}

struct SettingsStatusBadge: View {
    let title: String
    let color: Color

    var body: some View {
        Text(title)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(color.opacity(0.14), in: Capsule())
    }
}
