#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./build.sh <version> [--build-number N] [--skip-tests] [--local]

Build GitHub release artifacts for MacAssistant.

Arguments:
  <version>            Release version, for example: 0.1.0

Options:
  --build-number N     CFBundleVersion / build number (default: 1)
  --skip-tests         Skip the Debug test pass before archiving
  --local              Build local artifacts without Developer ID signing or notarization
  -h, --help           Show this help text

Environment variables for public releases:
  DEVELOPMENT_TEAM         Apple Developer Team ID
  DEVELOPER_ID_APPLICATION Exact "Developer ID Application: ..." identity name
  NOTARY_PROFILE           notarytool keychain profile name
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  local command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || die "missing required command: $command_name"
}

log() {
  printf '==> %s\n' "$*"
}

notarize() {
  local artifact="$1"
  log "Submitting $(basename "$artifact") for notarization"
  xcrun notarytool submit "$artifact" --keychain-profile "$NOTARY_PROFILE" --wait
}

verify_identity() {
  security find-identity -v -p codesigning | grep -Fq "$DEVELOPER_ID_APPLICATION" \
    || die "Developer ID identity not found: $DEVELOPER_ID_APPLICATION"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$SCRIPT_DIR/MacAssistant.xcodeproj"
SCHEME="MacAssistant"
APP_NAME="MacAssistant"

VERSION=""
BUILD_NUMBER="1"
SKIP_TESTS=0
LOCAL_MODE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build-number)
      [ "$#" -ge 2 ] || die "--build-number requires a value"
      BUILD_NUMBER="$2"
      shift 2
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --local)
      LOCAL_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [ -n "$VERSION" ]; then
        die "multiple version arguments provided"
      fi
      VERSION="$1"
      shift
      ;;
  esac
done

[ -n "$VERSION" ] || {
  usage
  exit 1
}

case "$VERSION" in
  *[!0-9A-Za-z.+-]*|'')
    die "version must use only letters, digits, dots, plus signs, and hyphens"
    ;;
esac

case "$BUILD_NUMBER" in
  *[!0-9]*|'')
    die "build number must be a positive integer"
    ;;
esac

[ "$BUILD_NUMBER" -ge 1 ] || die "build number must be a positive integer"

require_command xcodegen
require_command xcodebuild
require_command ditto
require_command hdiutil
require_command xcrun
require_command codesign
require_command spctl
require_command shasum
require_command security

if [ "$LOCAL_MODE" -eq 0 ]; then
  : "${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM for public releases}"
  : "${DEVELOPER_ID_APPLICATION:?set DEVELOPER_ID_APPLICATION for public releases}"
  : "${NOTARY_PROFILE:?set NOTARY_PROFILE for public releases}"
  xcrun --find notarytool >/dev/null 2>&1 || die "xcrun could not find notarytool"
  verify_identity
fi

DIST_DIR="$SCRIPT_DIR/dist/$VERSION"
TMP_DIR="$DIST_DIR/tmp"
ARCHIVE_PATH="$DIST_DIR/$APP_NAME.xcarchive"
ARCHIVED_APP_PATH="$ARCHIVE_PATH/Products/Applications/$APP_NAME.app"
APP_PATH="$DIST_DIR/$APP_NAME.app"
ZIP_PATH="$DIST_DIR/$APP_NAME-$VERSION.zip"
DMG_PATH="$DIST_DIR/$APP_NAME-$VERSION.dmg"
DMG_STAGING_DIR="$TMP_DIR/dmg-staging"
CHECKSUMS_PATH="$DIST_DIR/SHA256SUMS.txt"

rm -rf "$DIST_DIR"
mkdir -p "$TMP_DIR"

trap 'rm -rf "$TMP_DIR"' EXIT

log "Generating Xcode project"
(
  cd "$SCRIPT_DIR"
  xcodegen generate
)

if [ "$SKIP_TESTS" -eq 0 ]; then
  log "Running Debug tests"
  xcodebuild \
    -project "$PROJECT_PATH" \
    -scheme "$SCHEME" \
    -configuration Debug \
    test
fi

ARCHIVE_ARGS=(
  -project "$PROJECT_PATH"
  -scheme "$SCHEME"
  -configuration Release
  -archivePath "$ARCHIVE_PATH"
  archive
  MARKETING_VERSION="$VERSION"
  CURRENT_PROJECT_VERSION="$BUILD_NUMBER"
)

if [ "$LOCAL_MODE" -eq 0 ]; then
  ARCHIVE_ARGS+=(
    DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM"
    CODE_SIGN_STYLE=Manual
    CODE_SIGN_IDENTITY="$DEVELOPER_ID_APPLICATION"
  )
fi

log "Archiving $APP_NAME $VERSION ($BUILD_NUMBER)"
xcodebuild "${ARCHIVE_ARGS[@]}"

[ -d "$ARCHIVED_APP_PATH" ] || die "archive did not produce $ARCHIVED_APP_PATH"

log "Copying app bundle to dist/"
ditto "$ARCHIVED_APP_PATH" "$APP_PATH"

if [ "$LOCAL_MODE" -eq 0 ]; then
  log "Verifying signed app bundle"
  codesign --verify --deep --strict --verbose=2 "$APP_PATH"
else
  log "Ad-hoc signing local app bundle"
  codesign --force --deep --sign - --timestamp=none "$APP_PATH"
fi

log "Creating ZIP artifact"
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

if [ "$LOCAL_MODE" -eq 0 ]; then
  notarize "$ZIP_PATH"
  log "Stapling notarization ticket to app"
  xcrun stapler staple "$APP_PATH"
fi

log "Preparing DMG staging directory"
mkdir -p "$DMG_STAGING_DIR"
ditto "$APP_PATH" "$DMG_STAGING_DIR/$APP_NAME.app"
ln -s /Applications "$DMG_STAGING_DIR/Applications"

log "Creating DMG artifact"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$DMG_STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

if [ "$LOCAL_MODE" -eq 0 ]; then
  notarize "$DMG_PATH"
  log "Stapling notarization ticket to DMG"
  xcrun stapler staple "$DMG_PATH"
fi

log "Writing SHA256 checksums"
(
  cd "$DIST_DIR"
  shasum -a 256 "$(basename "$ZIP_PATH")" "$(basename "$DMG_PATH")" > "$(basename "$CHECKSUMS_PATH")"
)

if [ "$LOCAL_MODE" -eq 0 ]; then
  log "Running Gatekeeper verification"
  spctl -a -vv -t exec "$APP_PATH"
  spctl -a -vv -t open "$DMG_PATH"
else
  log "Skipping Gatekeeper verification in local mode"
fi

cat <<EOF

Artifacts ready in: $DIST_DIR
  App:       $APP_PATH
  ZIP:       $ZIP_PATH
  DMG:       $DMG_PATH
  Checksums: $CHECKSUMS_PATH

Upload the DMG to GitHub Releases as the primary asset.
Upload the ZIP as a fallback download.
EOF
