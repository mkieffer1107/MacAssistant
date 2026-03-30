import SwiftUI

struct LoadingView: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(spacing: 28) {
            VStack(spacing: 14) {
                Circle()
                    .fill(.white.opacity(0.10))
                    .frame(width: 72, height: 72)
                    .overlay {
                        ProgressView()
                            .controlSize(.large)
                            .tint(AppTheme.accent)
                            .scaleEffect(1.15)
                    }

                VStack(spacing: 6) {
                    Text(model.bootstrapStage)
                        .font(.title2.weight(.semibold))
                    Text(model.statusText)
                        .font(.body)
                        .foregroundStyle(AppTheme.secondaryText)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 440)
                }
            }

            if !model.bootstrapMessages.isEmpty {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(model.bootstrapMessages.enumerated()), id: \.offset) { index, message in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: index == model.bootstrapMessages.count - 1 ? "arrow.triangle.2.circlepath" : "checkmark.circle.fill")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(index == model.bootstrapMessages.count - 1 ? AppTheme.accent : AppTheme.success)
                                .frame(width: 18, height: 18)

                            Text(message)
                                .font(.system(size: 13))
                                .foregroundStyle(index == model.bootstrapMessages.count - 1 ? Color.primary : AppTheme.secondaryText)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .padding(.vertical, 12)

                        if index < model.bootstrapMessages.count - 1 {
                            Divider()
                                .overlay(AppTheme.separator)
                        }
                    }
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 8)
                .frame(maxWidth: 500, alignment: .leading)
                .sectionSurface(cornerRadius: 24, material: .ultraThinMaterial, shadowOpacity: 0.05)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.clear)
        .textSelection(.enabled)
    }
}
