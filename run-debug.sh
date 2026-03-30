#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DERIVED_DATA="${DERIVED_DATA:-$ROOT_DIR/.derivedData}"
PROJECT="$ROOT_DIR/MacAssistant.xcodeproj"
SCHEME="MacAssistant"
CONFIGURATION="Debug"
APP_PATH="$DERIVED_DATA/Build/Products/$CONFIGURATION/MacAssistant.app"
RUN_TESTS=1

usage() {
  echo "Usage: $0 [--skip-tests]" >&2
}

for arg in "$@"; do
  case "$arg" in
    --skip-tests)
      RUN_TESTS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

xcodegen generate

xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$DERIVED_DATA" \
  build

if [[ "$RUN_TESTS" -eq 1 ]]; then
  xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -derivedDataPath "$DERIVED_DATA" \
    -destination 'platform=macOS' \
    test
fi

pkill -x MacAssistant || true
open -n "$APP_PATH"
