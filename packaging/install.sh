#!/bin/sh
# Strata Terminal: one-line installer for Linux and macOS.
#
#     curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
#
# What it does, in order (and nothing else):
#   1. finds out which computer this is (system, processor, graphics card) so it can fetch the right archive;
#   2. downloads the Terminal archive named in the release folder's install-manifest.txt and checks its SHA-256 against it;
#   3. unpacks it into  ${XDG_DATA_HOME:-~/.local/share}/strata/terminal/<version>  (no administrator rights, nothing outside your home);
#   4. checks the release's signed feed with the Terminal it just unpacked (`strata update --check`); if that check fails, it removes
#      what it unpacked and stops;
#   5. puts a `strata` link in ~/.local/bin and prints:  Installed. Run: strata launch
#
# The engine (the part that talks to your graphics card) and the model are NOT part of this: `strata launch` fetches the engine for
# your card the first time you choose a model, from the same release folder, signed.
#
# <feed> is where the release folder is hosted: GitHub Releases of Niko1221/Strata (the "latest" release's files), built in as the default.
# Set STRATA_FEED to another folder (a URL) to use that instead, for example a local test feed:  STRATA_FEED=http://127.0.0.1:8099 sh install.sh
#
# Settings (all optional): STRATA_FEED, STRATA_DATA_DIR (default ~/.local/share/strata), STRATA_BIN_DIR (default ~/.local/bin),
# STRATA_INSTALL_TEST_TRUST=1 (a test feed: also fetch <feed>/test-trust.json and trust only that key for the check in step 4).
set -eu

FEED="${STRATA_FEED:-https://github.com/Niko1221/Strata/releases/latest/download}"
FEED="${FEED%/}"
DATA_DIR="${STRATA_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/strata}"
BIN_DIR="${STRATA_BIN_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'strata installer: %s\n' "$*" >&2; exit 1; }

[ -n "$FEED" ] || die "this installer does not know where the Strata release folder is. Set STRATA_FEED to its address and run it again, for example: STRATA_FEED=https://example.org/strata sh install.sh"

# ---- tools
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q -O "$2" "$1"; }
else
  die "neither curl nor wget is installed, so nothing can be downloaded. Install one (for example: sudo apt install curl) and run this again."
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  die "no sha256sum or shasum was found, so the download cannot be checked. Install coreutils and run this again."
fi
command -v tar >/dev/null 2>&1 || die "tar is not installed; install it and run this again."

# ---- 1. this computer
os=$(uname -s); arch=$(uname -m)
case "$os-$arch" in
  Linux-x86_64|Linux-amd64) platform=linux-x86_64 ;;
  Darwin-arm64|Darwin-aarch64) platform=macos-arm64 ;;
  Linux-aarch64|Linux-arm64) die "this computer ($os $arch) has an ARM processor. Strata needs an NVIDIA or AMD graphics card and has no engine for ARM Linux, so it could not run a model here. Use a Linux PC with an Intel or AMD processor (x86_64) and one of those cards." ;;
  Darwin-x86_64) die "this Mac has an Intel processor. Strata for Mac needs Apple silicon (M1 or newer), so it could not run a model here." ;;
  *) die "this computer ($os $arch) has no Strata Terminal. Supported: Linux x86_64 with an NVIDIA or AMD card, macOS on Apple silicon (Windows: use install.ps1)." ;;
esac
gpu=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  [ -n "$gpu" ] && gpu="NVIDIA $gpu"
fi
if [ -z "$gpu" ] && [ "$os" = Linux ] && command -v lspci >/dev/null 2>&1; then
  gpu=$(lspci 2>/dev/null | grep -iE 'vga|3d|display' | head -1 | sed 's/^[^:]*: *//' || true)
fi
if [ -z "$gpu" ] && [ "$os" = Darwin ]; then
  gpu=$(system_profiler SPDisplaysDataType 2>/dev/null | sed -n 's/^ *Chipset Model: //p' | head -1 || true)
fi
say "Strata Terminal installer"
say "  This computer: $os $arch ($platform)${gpu:+, graphics: $gpu}"
say "  Release folder: $FEED"

# ---- 2. download and check
work=$(mktemp -d "${TMPDIR:-/tmp}/strata-install.XXXXXX") || die "cannot make a temporary folder"
trap 'rm -rf "$work"' EXIT INT TERM
fetch "$FEED/install-manifest.txt" "$work/manifest.txt" || die "could not read $FEED/install-manifest.txt. Check the address and your internet connection."
line=$(awk -v p="$platform" '$1 == p { print; exit }' "$work/manifest.txt")
[ -n "$line" ] || die "this release has no Terminal for $platform."
set -- $line
version=$2; rel=$3; want=$4
say "  Terminal $version ($(( ${5:-0} / 1048576 )) MB)"
say "Downloading…"
fetch "$FEED/$rel" "$work/terminal.tar.gz" || die "the download failed. Check your internet connection and run this again; nothing was installed."
got=$(sha "$work/terminal.tar.gz")
[ "$got" = "$want" ] || die "the download does not match the release's checksum (got $got, wanted $want). It was not installed; try again, and if it repeats the release folder is damaged."
say "  Checksum ok."

# ---- 3. unpack
dest="$DATA_DIR/terminal/$version"
mkdir -p "$DATA_DIR/terminal" "$BIN_DIR"
rm -rf "$dest.incoming"; mkdir -p "$dest.incoming"
tar -xzf "$work/terminal.tar.gz" -C "$dest.incoming" || die "the archive could not be unpacked (is the disk full?). Nothing was installed."
if [ ! -f "$dest.incoming/bin/strata" ]; then                  # an archive with one top folder (a terminal archive instead of the feed's control package)
  inner=$(ls -d "$dest.incoming"/*/ 2>/dev/null | head -1)
  if [ -n "$inner" ] && [ -f "${inner}bin/strata" ]; then
    mv "${inner%/}" "$dest.inner" && rm -rf "$dest.incoming" && mv "$dest.inner" "$dest.incoming"
  fi
fi
[ -f "$dest.incoming/bin/strata" ] || { rm -rf "$dest.incoming"; die "the archive does not look like a Strata Terminal (no bin/strata). Nothing was installed."; }
chmod +x "$dest.incoming/bin/strata" 2>/dev/null || true

# ---- 4. the signed feed, checked with the Terminal just unpacked
if [ "${STRATA_INSTALL_TEST_TRUST:-}" = 1 ]; then
  fetch "$FEED/test-trust.json" "$work/test-trust.json" || { rm -rf "$dest.incoming"; die "the test feed has no test-trust.json"; }
  STRATA_UPDATE_TRUSTED_KEYS="$work/test-trust.json"; export STRATA_UPDATE_TRUSTED_KEYS
fi
say "Checking the release's signature…"
if ! "$dest.incoming/bin/strata" update --check --no-engine --feed "$FEED" >"$work/check.txt" 2>&1; then
  sed -e 's/^/    /' "$work/check.txt" >&2
  rm -rf "$dest.incoming"
  die "the release folder's signed feed could not be verified, so nothing was installed. The reason is printed above; if it names a newer feed already accepted here, use the newest release folder, or follow the step it gives. If you trust this source and see no such reason, ask its owner to publish a signed feed."
fi
say "  Signature ok."

# ---- 5. in place
rm -rf "$dest"
mv "$dest.incoming" "$dest"
ln -sfn "$dest" "$DATA_DIR/terminal/current"
ln -sf "$DATA_DIR/terminal/current/bin/strata" "$BIN_DIR/strata"
case ":$PATH:" in
  *":$BIN_DIR:"*) say "Installed. Run: strata launch" ;;
  *) say "Installed. Run: $BIN_DIR/strata launch"
     say "  (To type just 'strata', add this line to your shell profile: export PATH=\"$BIN_DIR:\$PATH\")" ;;
esac
