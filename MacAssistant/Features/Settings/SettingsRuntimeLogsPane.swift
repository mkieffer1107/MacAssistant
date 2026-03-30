import SwiftUI

struct SettingsRuntimeLogsPane: View {
    let snapshot: AppModel.SettingsRuntimeLogsSnapshot
    let copyRuntimeLogs: () -> Void

    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        formatter.zeroPadsFractionDigits = false
        return formatter
    }()

    var body: some View {
        let hasLogs = !snapshot.entries.isEmpty

        return VStack(spacing: 0) {
            SettingsSectionCard {
                SettingsSectionHeader(
                    systemImage: "terminal",
                    title: "Runtime Logs",
                    subtitle: "Recent runtime and microphone diagnostics."
                ) {
                    VStack(alignment: .trailing, spacing: 6) {
                        Button("Copy Recent", action: copyRuntimeLogs)
                            .buttonStyle(.bordered)
                            .controlSize(.regular)
                            .disabled(!hasLogs)

                        Text(logRetentionSummary)
                            .font(.caption)
                            .foregroundStyle(AppTheme.secondaryText)
                    }
                }

                ScrollView {
                    if hasLogs {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(snapshot.entries) { entry in
                                Text(verbatim: entry.text)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(Color.primary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .textSelection(.enabled)
                                    .id(entry.id)
                            }
                        }
                    } else {
                        Text("No runtime logs yet.")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(AppTheme.secondaryText)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .frame(maxWidth: .infinity, minHeight: 260, maxHeight: .infinity, alignment: .topLeading)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(AppTheme.actionCardFill)
                )
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(.vertical, 6)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var logRetentionSummary: String {
        if snapshot.lineCount == 0 {
            return "0 lines retained"
        }
        return "\(snapshot.lineCount) lines • \(Self.byteFormatter.string(fromByteCount: Int64(snapshot.byteCount))) retained"
    }
}
