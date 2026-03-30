# MacAssistant

Native macOS Tahoe 26 SwiftUI app for a local voice-first automation assistant.

## What is implemented

- Small always-on-top floating chat window with setup, loading, chat, and settings flows.
- Single-thread conversation UI with:
  - typed messages
  - tap-to-start / tap-to-stop voice capture
  - streaming voice draft transcription
  - streaming assistant text
  - collapsed tool-call rows with approve / deny actions
  - global stop control
- Bundled helper runtime launched over NDJSON stdio from the app.
- Model management for:
  - `Voice Pack` (`stt_model` + `tts_model`)
  - `Agent Model` (`agent_model`)
- Runtime host wiring for:
  - Hugging Face model download and deletion
  - MLX warm/load paths for the generic agent, speech input, and speech output model IDs
  - MCP tool execution through `@steipete/macos-automator-mcp`
  - safe auto-run vs approval-required tool policy
  - runtime manifest and cache storage under `~/Library/Application Support/MacAssistant/`

## Current constraints

- The Swift app, IPC contract, and bundled runtime are implemented and buildable.
- The helper now uses the real Python-side libraries and MCP client, but this workspace has only been smoke-tested through bootstrap and packaging. Large-model download and end-to-end inference were not exercised in this session.
- The runtime currently expects `uv` and `bun` to be available on the machine. The app bootstraps its Python environment into Application Support on first run.

## Build

```bash
DERIVED_DATA="$PWD/.derivedData"

xcodegen generate
xcodebuild \
  -project MacAssistant.xcodeproj \
  -scheme MacAssistant \
  -configuration Debug \
  -derivedDataPath "$DERIVED_DATA" \
  build
xcodebuild \
  -project MacAssistant.xcodeproj \
  -scheme MacAssistant \
  -configuration Debug \
  -derivedDataPath "$DERIVED_DATA" \
  -destination 'platform=macOS' \
  test
```

To build, test, stop any existing app instance, and launch the exact Debug app you just built:

```bash
./run-debug.sh
```

The script uses the repo-local `.derivedData/` directory so the launch path is stable. Do not use `open "$(find ~/Library/Developer/Xcode/DerivedData ...)"` for Debug runs, because it can reopen an older app bundle from a different DerivedData folder.

## Release

Use the repo-root build script to generate GitHub release artifacts:

```bash
./build.sh 0.1.0 --local
```

That command creates a local test release in `dist/0.1.0/` with:

- `MacAssistant-0.1.0.dmg`: primary GitHub Release asset
- `MacAssistant-0.1.0.zip`: fallback download with the app bundle
- `SHA256SUMS.txt`: checksums for the release artifacts

For a public signed and notarized release, install a `Developer ID Application` certificate first, store a `notarytool` profile, then run:

```bash
export DEVELOPMENT_TEAM=YOURTEAMID
export DEVELOPER_ID_APPLICATION="Developer ID Application: Your Name (YOURTEAMID)"
export NOTARY_PROFILE=macassistant-notary
./build.sh 0.1.0 --build-number 1
```

To configure the `notarytool` profile:

```bash
xcrun notarytool store-credentials macassistant-notary \
  --apple-id YOUR_APPLE_ID \
  --team-id YOURTEAMID \
  --password YOUR_APP_SPECIFIC_PASSWORD
```

If this Mac only has an `Apple Development` identity installed, use `--local` until the `Developer ID Application` certificate is available.

The script runs `xcodegen generate`, runs tests unless `--skip-tests` is passed, archives the app in `Release`, and produces the uploadable DMG and ZIP for GitHub Releases. The DMG should be the primary release file on GitHub; a `.pkg` is not necessary for this app.

The app release is still not fully self-contained. End users must have `uv` and `bun` available on their Mac for the bundled runtime bootstrap to work.

To open the project in Xcode:
```bash
open MacAssistant.xcodeproj
```

To launch the app:
```bash
open -n "$PWD/.derivedData/Build/Products/Debug/MacAssistant.app"
```


## Layout

- `MacAssistant/`: SwiftUI app shell, features, models, services, and window styling
- `Runtime/`: bundled runtime bootstrap script, Python host, and Python requirements
- `MacAssistantTests/`: unit tests for installable aggregation and tool safety state
- `build.sh`: release packaging entrypoint for DMG and ZIP artifacts
- `project.yml`: XcodeGen source of truth
