# AGENTS.md

This file applies to the entire repository.

## Project Overview

- `MacAssistant/App/` contains the app entry point, root phase routing, and window lifecycle hooks.
- `MacAssistant/Features/` contains the setup, loading, chat, and settings SwiftUI screens.
- `MacAssistant/Models/` contains conversation, model/install state, and JSON bridge types.
- `MacAssistant/Services/` contains app state, runtime IPC, runtime launching, microphone capture, and audio playback.
- `MacAssistant/Utilities/` contains shared theme and window helper code.
- `Runtime/` contains the bundled Python runtime launched over NDJSON stdio.
- `Runtime/tests/` contains Python runtime tests.
- `MacAssistantTests/` contains Swift Testing unit tests.
- `project.yml` is the XcodeGen source of truth.
- `MacAssistant.xcodeproj` and `dist/` are generated artifacts.

## Working Rules

- Prefer changing source files over generated outputs. If the project structure, copied runtime resources, or build settings change, edit `project.yml` and regenerate the Xcode project.
- If you change the runtime protocol, update both sides together:
  - `MacAssistant/Services/RuntimeProtocol.swift`
  - `Runtime/agent_runtime_host.py`
- If you change turn lifecycle, send/receive flow, tool approval state, speech state, or persisted settings, inspect `MacAssistant/Services/AppModel.swift` first.
- If you change runtime process launch, environment propagation, stdio handling, or runtime log filtering, inspect `MacAssistant/Services/AgentRuntimeClient.swift` first.
- If you change audio recording or playback behavior, inspect these files together:
  - `MacAssistant/Services/MicrophoneCaptureService.swift`
  - `MacAssistant/Services/AudioPlaybackService.swift`
- If you change live voice transcription or recording controls, inspect these files together:
  - `MacAssistant/Services/AppModel.swift`
  - `MacAssistant/Services/MicrophoneCaptureService.swift`
  - `MacAssistant/Features/Chat/ChatView.swift`
  - `MacAssistantTests/MacAssistantTests.swift`
- If you change phase routing, drag-and-drop entry points, or window behavior, inspect these files first:
  - `MacAssistant/App/RootView.swift`
  - `MacAssistant/App/MacAssistantApp.swift`
  - `MacAssistant/Utilities/WindowAccessor.swift`
- If you change chat layout or message rendering, inspect `MacAssistant/Features/Chat/ChatView.swift` first.
- If you change the model install/setup flow, inspect `MacAssistant/Features/Setup/SetupView.swift` first.
- If you change settings UI or debugging surfaces, inspect `MacAssistant/Features/Settings/SettingsView.swift` first.
- If you change app look and feel, inspect `MacAssistant/Utilities/AppTheme.swift` first.
- Do not hand-edit `dist/` unless release packaging is part of the task. Rebuild it through `./build.sh`.
- Do not hand-edit `MacAssistant.xcodeproj` unless regeneration is impossible for the task. Prefer `xcodegen generate`.
- For local Debug runs, prefer `./run-debug.sh` or pass `-derivedDataPath "$PWD/.derivedData"` to `xcodebuild` and launch `$PWD/.derivedData/Build/Products/Debug/MacAssistant.app` directly. Do not use `open "$(find ~/Library/Developer/Xcode/DerivedData ...)"`, because it can reopen a stale app bundle from a different DerivedData folder.
- Do not delete downloaded models or the user's `~/Library/Application Support/MacAssistant/` runtime state unless the user explicitly asks for that.

## Runtime Notes

- The app bootstraps a private Python environment through `Runtime/AgentRuntimeHost.sh`.
- The Xcode target copies these runtime resources from the repo into the app bundle at build time:
  - `Runtime/AgentRuntimeHost.sh`
  - `Runtime/agent_runtime_host.py`
  - `Runtime/requirements.txt`
- The Swift app and Python host communicate over NDJSON stdio.
- Runtime state, downloaded models, caches, and logs live under `~/Library/Application Support/MacAssistant/`.
- Changes to `Runtime/requirements.txt` trigger a runtime refresh through the `requirements.stamp` hash on next launch.
- The runtime expects `uv` and `bun` to be available on the machine.
- Tool execution is routed through `@steipete/macos-automator-mcp`.
- The main installables are `voice_pack` (`stt_model` + `tts_model`) and `agent_model`.
- The agent path is multimodal. Text-only chat still runs through the same agent model unless routing explicitly shortcuts planner work.
- Live voice capture is session-scoped. `MicrophoneCaptureService` creates a fresh capture session/converter per recording and should be fully reset on every stop or failure.
- Realtime STT is streamed through the existing NDJSON events. The app should open `start_recording` only after the first captured mic chunk, stream `append_audio_chunk` while speaking, use cumulative `transcript_delta` text for the draft bubble, and reserve `transcript_final` for commit/finalization.
- Recording controls are intentionally asymmetric:
  - send during recording finalizes and submits the live voice draft
  - the mic-side stop/discard path cancels the draft and should leave the mic immediately reusable
  - empty voice transcripts without an attachment should remove the draft instead of converting it into a text message

## Validation

Run the smallest relevant checks for the change:

- Swift app tests:
  - `xcodebuild -project MacAssistant.xcodeproj -scheme MacAssistant -derivedDataPath "$PWD/.derivedData" -destination 'platform=macOS' test`
- Swift app build:
  - `xcodebuild -project MacAssistant.xcodeproj -scheme MacAssistant -configuration Debug -derivedDataPath "$PWD/.derivedData" build`
- Local Debug build, test, and launch:
  - `./run-debug.sh`
- Python runtime tests:
  - `python3 -m unittest Runtime/tests/test_agent_runtime_host.py`
- Voice/STT-focused runtime streaming checks:
  - `python3 -m unittest Runtime.tests.test_agent_runtime_host.BufferedPreviewSessionTests.test_continuous_audio_emits_partial_before_finish`
  - `python3 -m unittest Runtime.tests.test_agent_runtime_host.RuntimeHostStreamingTests.test_stop_recording_uses_live_session_when_present`
  - `python3 -m unittest Runtime.tests.test_agent_runtime_host.RuntimeHostStreamingTests.test_fallback_session_still_streams_preview_and_finalizes_batch`
- Regenerate the Xcode project when `project.yml` or copied runtime resources change:
  - `xcodegen generate`
- Local release packaging smoke test:
  - `./build.sh 0.1.0 --local`

## Release Packaging

- Local release build:
  - `./build.sh 0.1.0 --local`
- Public signed build requires the environment variables documented in `README.md` and `build.sh`.
- If source runtime files change and you are asked to update shipped artifacts, rebuild so these bundled files stay in sync:
  - `dist/<version>/MacAssistant.app/Contents/Resources/AgentRuntimeHost.sh`
  - `dist/<version>/MacAssistant.app/Contents/Resources/agent_runtime_host.py`
  - `dist/<version>/MacAssistant.app/Contents/Resources/requirements.txt`
- `build.sh` regenerates the Xcode project, runs tests unless `--skip-tests` is passed, archives the Release app, and produces the DMG, ZIP, and checksum outputs.

## High-Value Files

- `MacAssistant/App/RootView.swift`: phase routing, drag-and-drop entry point, shell surface
- `MacAssistant/App/MacAssistantApp.swift`: app lifecycle, window scene, startup/shutdown hooks
- `MacAssistant/Services/AppModel.swift`: app state, turn lifecycle, model/install state
- `MacAssistantTests/MacAssistantTests.swift`: Swift regression coverage, including live voice draft and recording recovery cases
- `MacAssistant/Services/AgentRuntimeClient.swift`: runtime launch, environment, NDJSON IO, launch logging
- `MacAssistant/Features/Chat/ChatView.swift`: chat UI, composer, attachment preview
- `MacAssistant/Features/Setup/SetupView.swift`: model install flow and runtime status cards
- `MacAssistant/Features/Settings/SettingsView.swift`: settings toggles, model management, runtime logs
- `MacAssistant/Models/ConversationModels.swift`: user/assistant message models
- `MacAssistant/Models/ModelState.swift`: installable and underlying model state mapping
- `MacAssistant/Services/RuntimeProtocol.swift`: app/runtime IPC payloads
- `MacAssistant/Services/MicrophoneCaptureService.swift`: live microphone capture and PCM conversion
- `MacAssistant/Services/AudioPlaybackService.swift`: assistant speech playback
- `MacAssistant/Utilities/AppTheme.swift`: shared visual styling, backdrops, and surfaces
- `MacAssistant/Utilities/WindowAccessor.swift`: floating window configuration and NSWindow integration
- `Runtime/agent_runtime_host.py`: model loading, routing, tool planning, multimodal generation
- `Runtime/AgentRuntimeHost.sh`: Python bootstrap and dependency refresh
- `Runtime/tests/test_agent_runtime_host.py`: Python runtime behavior coverage
- `build.sh`: archive, ZIP, DMG, and release artifact generation
- `project.yml`: target definitions, copied runtime resources, and scheme configuration
