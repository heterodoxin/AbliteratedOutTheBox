#!/usr/bin/env bash
# Fetch and verify the private Node runtime the desktop bundle ships.
#
# The App Host is an ordinary Node program, and the packaging requirement is that an end user
# supplies no runtime of their own. The desktop bundle therefore carries a pinned standalone Node
# beside the host.
#
# It is NOT committed. It is a ~117 MB platform binary: committing it would bloat every clone, it
# cannot be reviewed in a diff, and it is not source. This script is the reproducible substitute —
# it downloads the official build for the target and refuses to install anything whose SHA-256 does
# not match the recorded digest.
#
# Usage:
#   apps/desktop/scripts/fetch-node.sh                 # host platform
#   apps/desktop/scripts/fetch-node.sh --target linux-x64
#   apps/desktop/scripts/fetch-node.sh --check         # verify what is already installed
#
# The Node version must satisfy the App Host's own `engines.node` (>= 24.12.0).
set -euo pipefail

NODE_VERSION="${STRATA_NODE_VERSION:-24.12.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../src-tauri/resources/node"

# Published SHA-256 of each official nodejs.org archive, recorded so a tampered or truncated
# download is refused rather than shipped. Add a row when bumping the version.
# Taken from nodejs.org's own SHASUMS256.txt for v24.12.0, which is published over HTTPS beside
# the archives. Recorded here so the fetch is verified against a value that does not come from the
# same response as the payload.
digest_for() {
  case "$1" in
    darwin-arm64) echo "319f221adc5e44ff0ed57e8a441b2284f02b8dc6fc87b8eb92a6a93643fd8080" ;;
    linux-x64)    echo "6159227e0af7d7c3c6bb2fa900452b04a6cb8841a702a79acc613209d70b04d0" ;;
    linux-arm64)  echo "9b2a2eeb98a8eb37361224e2a1d060300ad2dd143af58dfdb16de785df0f1228" ;;
    win-x64)      echo "9c125f61ae947b52e779095830f9cac267846a043ef7192183c84016aaad2812" ;;
    *)            echo "" ;;
  esac
}

target() {
  if [ "${1:-}" = "--target" ]; then echo "${2:?--target needs a value}"; return; fi
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) echo darwin-arm64 ;;
    Linux-x86_64) echo linux-x64 ;;
    Linux-aarch64) echo linux-arm64 ;;
    MINGW*|MSYS*|CYGWIN*) echo win-x64 ;;
    *) echo "unsupported host: $(uname -s)-$(uname -m)" >&2; exit 2 ;;
  esac
}

TARGET="$(target "$@")"
EXE="node"
case "$TARGET" in win-*) EXE="node.exe" ;; esac
INSTALLED="$DEST/$EXE"

if [ "${1:-}" = "--check" ]; then
  if [ ! -x "$INSTALLED" ]; then echo "fetch-node: not installed at $INSTALLED" >&2; exit 1; fi
  got="$("$INSTALLED" --version)"
  echo "fetch-node: $INSTALLED is $got"
  case "$got" in "v$NODE_VERSION") exit 0 ;; *) echo "fetch-node: expected v$NODE_VERSION" >&2; exit 1 ;; esac
fi

# The official archive for this target.
case "$TARGET" in
  darwin-arm64) ARCHIVE="node-v$NODE_VERSION-darwin-arm64.tar.gz" ;;
  darwin-x64)   ARCHIVE="node-v$NODE_VERSION-darwin-x64.tar.gz" ;;
  linux-x64)    ARCHIVE="node-v$NODE_VERSION-linux-x64.tar.gz" ;;
  linux-arm64)  ARCHIVE="node-v$NODE_VERSION-linux-arm64.tar.gz" ;;
  win-x64)      ARCHIVE="node-v$NODE_VERSION-win-x64.zip" ;;
  *) echo "fetch-node: no archive known for $TARGET" >&2; exit 2 ;;
esac
URL="https://nodejs.org/dist/v$NODE_VERSION/$ARCHIVE"

EXPECTED="$(digest_for "$TARGET")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/strata-node.XXXXXX")"     # GNU mktemp needs the X template; BSD accepts it too
trap 'rm -rf "$WORK"' EXIT

echo "fetch-node: $URL"
curl -fsSL -o "$WORK/$ARCHIVE" "$URL"

ACTUAL="$(shasum -a 256 "$WORK/$ARCHIVE" 2>/dev/null | cut -d' ' -f1 || sha256sum "$WORK/$ARCHIVE" | cut -d' ' -f1)"
if [ -n "$EXPECTED" ]; then
  if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "fetch-node: REFUSING — $ARCHIVE hashes $ACTUAL, expected $EXPECTED" >&2
    exit 1
  fi
  echo "fetch-node: sha256 verified against the recorded digest"
else
  # Without a recorded digest the download cannot be authenticated. Say so loudly and keep the
  # computed value so it can be recorded, rather than implying a verification that did not happen.
  echo "fetch-node: WARNING — no recorded digest for $TARGET; this download is NOT authenticated." >&2
  echo "fetch-node: computed sha256 $ACTUAL — record it in $(basename "${BASH_SOURCE[0]}") to enable verification." >&2
fi

mkdir -p "$DEST"
case "$ARCHIVE" in
  *.zip) unzip -q -j "$WORK/$ARCHIVE" "node-v$NODE_VERSION-win-x64/node.exe" -d "$DEST" ;;
  *)     tar xzf "$WORK/$ARCHIVE" -C "$WORK"
         cp "$WORK/node-v$NODE_VERSION-$TARGET/bin/node" "$DEST/node" ;;
esac
chmod +x "$DEST/$EXE"

echo "fetch-node: installed $("$DEST/$EXE" --version) at $DEST/$EXE"
