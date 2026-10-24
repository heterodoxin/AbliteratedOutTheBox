#!/usr/bin/env bash
# Build the Strata DMG without a Finder session.
#
# Tauri's default DMG step runs an AppleScript that asks Finder to arrange the window (icon
# positions, background). Finder AppleEvents time out in a non-GUI or sandboxed context — observed
# here as `Finder got an error: AppleEvent timed out. (-1712)` — so the default path cannot be
# relied on for an automated build. The upstream `create-dmg` script already supports exactly this
# case via `--skip-jenkins`, which produces a plain, correct DMG with the app and the Applications
# drop link and no cosmetic arrangement.
#
# Usage: scripts/make-dmg.sh [output.dmg]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP="$(cd "$HERE/.." && pwd)"
BUNDLE="$DESKTOP/src-tauri/target/release/bundle"
APP="$BUNDLE/macos/Strata.app"
VERSION="$(grep -m1 '^version = ' "$DESKTOP/src-tauri/Cargo.toml" | cut -d'"' -f2)"   # the one place the app version is written
OUT="${1:-$BUNDLE/dmg/Strata_${VERSION}_aarch64.dmg}"

if [ ! -d "$APP" ]; then
  echo "error: $APP does not exist; run 'cargo tauri build --bundles app' first" >&2
  exit 1
fi

WORK="$(mktemp -d -t strata-dmg)"
trap 'rm -rf "$WORK"' EXIT

# A staging folder is what create-dmg copies into the image: the app, and nothing else.
#
# The Applications drop link is created by `--app-drop-link` below. Adding one here as well makes
# create-dmg fail with "File exists", because it links into the volume it is assembling.
mkdir -p "$WORK/stage"
cp -R "$APP" "$WORK/stage/"

# Reuse Tauri's vendored create-dmg if the bundler left it behind, otherwise take it from the
# bundle cache. Building it yourself is not necessary and not expected.
SCRIPT="$BUNDLE/dmg/bundle_dmg.sh"
if [ ! -x "$SCRIPT" ]; then
  echo "error: $SCRIPT not found; run 'cargo tauri build --bundles dmg' once to vendor it" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

# --skip-jenkins: no Finder AppleScript. Everything else is the same code path Tauri uses.
"$SCRIPT" \
  --volname "Strata $VERSION" \
  --volicon "$BUNDLE/macos/Strata.app/Contents/Resources/Strata.icns" \
  --app-drop-link 400 200 \
  --skip-jenkins \
  --no-internet-enable \
  "$OUT" "$WORK/stage"

echo
echo "built: $OUT"
ls -lh "$OUT" | awk '{print "size:  " $5}'
shasum -a 256 "$OUT"
