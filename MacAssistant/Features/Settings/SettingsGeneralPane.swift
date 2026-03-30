import SwiftUI

struct SettingsGeneralPane: View {
    let snapshot: AppModel.SettingsGeneralSnapshot
    let defaultVoiceSelection: Binding<String>
    let inputDeviceSelection: Binding<String>
    let launchOnOpenSelection: Binding<Bool>
    let alwaysAcceptToolCallsSelection: Binding<Bool>
    let streamReplySpeechSelection: Binding<Bool>

    private let builtInVoicePresets: [(id: String, title: String)] = [
        ("casual_male", "Casual Male"),
        ("casual_female", "Casual Female"),
        ("cheerful_female", "Cheerful Female"),
        ("neutral_male", "Neutral Male"),
        ("neutral_female", "Neutral Female")
    ]

    var body: some View {
        ScrollView {
            SettingsSectionCard {
                SettingsSectionHeader(
                    systemImage: "slider.horizontal.3",
                    title: "General",
                    subtitle: "Voice defaults and automation behavior."
                )

                SettingsRow(
                    title: "Input Device",
                    subtitle: "Automatic prefers the built-in microphone when one is available.",
                    controlWidth: 320
                ) {
                    inputDevicePicker
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Default Voice",
                    subtitle: "Choose the voice used for spoken replies.",
                    controlWidth: 220
                ) {
                    defaultVoicePicker
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Launch on login",
                    subtitle: "Open MacAssistant automatically after you sign in."
                ) {
                    Toggle("", isOn: launchOnOpenSelection)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Always accept tool calls",
                    subtitle: "Skip approval prompts when the runtime proposes a tool call."
                ) {
                    Toggle("", isOn: alwaysAcceptToolCallsSelection)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Stream spoken replies",
                    subtitle: "Start speaking before the full reply finishes generating."
                ) {
                    Toggle("", isOn: streamReplySpeechSelection)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
            }
            .padding(.vertical, 6)
        }
        .scrollIndicators(.hidden)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var inputDevicePicker: some View {
        Picker("Input Device", selection: inputDeviceSelection) {
            Text("Automatic").tag("")
            ForEach(snapshot.availableInputDevices) { device in
                Text(deviceLabel(for: device))
                    .tag(device.uid)
            }
            if let unavailableUID = snapshot.unavailableSelectedInputDeviceUID {
                Text("Unavailable Device (\(unavailableUID))")
                    .tag(unavailableUID)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .controlSize(.regular)
    }

    private var defaultVoicePicker: some View {
        Picker("Default Voice", selection: defaultVoiceSelection) {
            ForEach(builtInVoicePresets, id: \.id) { voice in
                Text(voice.title)
                    .tag(voice.id)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .controlSize(.regular)
    }

    private func deviceLabel(for device: MicrophoneCaptureService.InputDevice) -> String {
        let suffix: String
        switch device.transport {
        case .builtIn:
            suffix = "Built-In"
        case .bluetooth:
            suffix = "Bluetooth"
        case .usb:
            suffix = "USB"
        case .aggregate:
            suffix = "Aggregate"
        case .virtual:
            suffix = "Virtual"
        case .unknown:
            suffix = "External"
        }
        return "\(device.name) (\(suffix))"
    }
}
