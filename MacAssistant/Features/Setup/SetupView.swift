import Foundation
import SwiftUI

struct SetupView: View {
    @Bindable var model: AppModel
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
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                runtimeStatusCard

                VStack(spacing: 14) {
                    ForEach(model.modelInstallables) { installable in
                        setupCard(installable)
                    }
                }
            }
            .padding(24)
        }
        .scrollIndicators(.hidden)
        .background(.clear)
        .textSelection(.enabled)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 16) {
            ZStack {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(.white.opacity(0.10))
                    .frame(width: 62, height: 62)
                Image(systemName: "apple.intelligence")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(AppTheme.accent)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Set Up MacAssistant")
                    .font(.largeTitle.weight(.semibold))
                Text("Install the local voice and agent models once. After that, the app will reopen and warm them automatically.")
                    .font(.body)
                    .foregroundStyle(AppTheme.secondaryText)
                    .frame(maxWidth: 520, alignment: .leading)
            }

            Spacer(minLength: 0)
        }
        .padding(24)
        .sectionSurface(cornerRadius: 28, material: .thinMaterial, shadowOpacity: 0.06)
    }

    private var runtimeStatusCard: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: "cpu")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(AppTheme.accent)
                .frame(width: 34, height: 34)
                .background(.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            VStack(alignment: .leading, spacing: 5) {
                Text(runtimeStatusTitle)
                    .font(.headline)
                Text(model.statusText)
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
                Text(runtimeStatusFootnote)
                    .font(.caption)
                    .foregroundStyle(AppTheme.tertiaryText)

                if let progress = model.activeDownloadProgress {
                    downloadProgressView(progress, compact: true)
                        .padding(.top, 6)
                }
            }

            Spacer(minLength: 0)
        }
        .frostedCard()
    }

    private var runtimeStatusTitle: String {
        let downloadingInstallables = model.modelInstallables.filter { $0.installState == .downloading }
        if downloadingInstallables.count == 1 {
            return "Downloading \(downloadingInstallables[0].title)"
        }
        if downloadingInstallables.count > 1 {
            return "Downloading Models"
        }

        switch model.phase {
        case .loading:
            return "Warming Models"
        case .ready:
            return "Runtime Ready"
        case .bootstrapping, .setup:
            return formattedRuntimeStage(model.bootstrapStage)
        }
    }

    private var runtimeStatusFootnote: String {
        if model.modelInstallables.contains(where: { $0.installState == .downloading }) {
            return "Downloads are in progress. You can queue installs from the cards below."
        }

        switch model.phase {
        case .loading:
            return "Installed models are warming so setup can finish automatically."
        case .ready:
            return "Core models are ready. Any remaining background warmup will continue automatically."
        case .bootstrapping, .setup:
            return "Use the install buttons below when you are ready."
        }
    }

    private func formattedRuntimeStage(_ stage: String) -> String {
        switch stage {
        case "Checking runtime prerequisites":
            return "Preparing Runtime"
        case "Connecting automation tools":
            return "Connecting Tools"
        case "Checking installed models":
            return "Checking Models"
        case "Runtime ready":
            return "Runtime Ready"
        default:
            return stage
        }
    }

    private func setupCard(_ installable: ModelInstallable) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: iconName(for: installable))
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(AppTheme.accent)
                    .frame(width: 40, height: 40)
                    .background(.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

                VStack(alignment: .leading, spacing: 4) {
                    Text(installable.title)
                        .font(.title3.weight(.semibold))
                    Text(installable.summary)
                        .font(.subheadline)
                        .foregroundStyle(AppTheme.secondaryText)
                }

                Spacer(minLength: 0)

                installStateBadge(installable.installState)
            }

            HStack {
                Label(installable.downloadSize, systemImage: "arrow.down.circle")
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.secondaryText)
                Spacer()
                if installable.installState == .installed {
                    Button("Delete") {
                        model.delete(installable)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                }
                if installable.installState != .downloading {
                    Button(installButtonTitle(for: installable.installState)) {
                        model.install(installable)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(AppTheme.accent)
                    .controlSize(.large)
                }
            }

            if let progress = model.downloadProgress(for: installable.id),
               installable.installState == .downloading {
                downloadProgressView(progress, compact: false)
            }

            if let lastError = installable.lastError {
                Label(lastError, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(AppTheme.destructive)
            }
        }
        .padding(20)
        .sectionSurface(cornerRadius: 24, material: .ultraThinMaterial, shadowOpacity: 0.05)
    }

    private func installStateBadge(_ state: ModelInstallState) -> some View {
        let color: Color = switch state {
        case .missing: AppTheme.warning
        case .downloading: AppTheme.accent
        case .installed: AppTheme.success
        case .failed: AppTheme.destructive
        }

        return Label(state.rawValue.capitalized, systemImage: state == .downloading ? "arrow.down.circle.fill" : "circle.fill")
            .labelStyle(.titleAndIcon)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(color.opacity(0.14), in: Capsule())
    }

    private func installButtonTitle(for state: ModelInstallState) -> String {
        switch state {
        case .missing:
            return "Install"
        case .downloading:
            return "Downloading..."
        case .installed:
            return "Reinstall"
        case .failed:
            return "Retry"
        }
    }

    private func iconName(for installable: ModelInstallable) -> String {
        switch installable.id {
        case "voice_pack":
            return "waveform.badge.mic"
        case "agent_model":
            return "brain.head.profile"
        default:
            return "shippingbox"
        }
    }

    private func downloadProgressView(_ progress: InstallableDownloadProgress, compact: Bool) -> some View {
        VStack(alignment: .leading, spacing: compact ? 6 : 8) {
            if let fractionCompleted = progress.fractionCompleted {
                ProgressView(value: fractionCompleted)
                    .progressViewStyle(.linear)
                    .tint(AppTheme.accent)
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
                    .tint(AppTheme.accent)
            }

            HStack(alignment: .firstTextBaseline) {
                Text(downloadProgressSummary(progress))
                    .font(compact ? .caption : .footnote)
                    .foregroundStyle(AppTheme.secondaryText)
                Spacer(minLength: 12)
                Text(downloadProgressTiming(progress))
                    .font(compact ? .caption : .footnote)
                    .foregroundStyle(AppTheme.tertiaryText)
            }
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
