#!/usr/bin/env bash
# Write a tester's README with the checksums and sizes of the files it names, computed from the files themselves.
#
#   apps/desktop/scripts/make-tester-readme.sh <template> <out.txt> --appimage <file name> --files-dir <dir> [--files-dir <dir> ...]
#
# Why: a README that carries a hand-pasted sha256 goes stale the moment the file is rebuilt (the Ubuntu 26.04 tester compared the file
# with the old build's hash). The template holds placeholders instead, and this script fills them from the real files:
#   @APPIMAGE@            the AppImage's file name
#   @APP_VERSION@         the app version, read from apps/desktop/src-tauri/Cargo.toml
#   @SHA256:<name>@       sha256 of the file <name> found in one of the --files-dir folders
#   @SIZE_GB:<name>@      its size in GB with one decimal
#   @INCLUDE:<file>@      (alone on a line) the contents of <file> from the template's folder, with the same placeholders
# It refuses to write the README when a named file is missing or a placeholder is left over, so a stale or empty hash cannot ship.
# The same check is done by `--verify <readme> --files-dir <dir>`: every "<64 hex>  <name>" or "sha256 <64 hex>" pair that names a file
# present in the folders must match that file.
set -euo pipefail

usage() { echo "usage: $0 <template> <out.txt> --appimage <name> --files-dir <dir> [--files-dir <dir> ...]" >&2; echo "       $0 --verify <readme> --files-dir <dir> [...]" >&2; exit 2; }

VERIFY=""; TEMPLATE=""; OUT=""; APPIMAGE=""; DIRS=()
if [ "${1:-}" = "--verify" ]; then VERIFY="${2:?readme}"; shift 2; else TEMPLATE="${1:?template}"; OUT="${2:?out}"; shift 2 || usage; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --appimage) APPIMAGE="$2"; shift 2 ;;
    --files-dir) DIRS+=("$2"); shift 2 ;;
    *) usage ;;
  esac
done
[ ${#DIRS[@]} -gt 0 ] || usage

find_file() { local d; for d in "${DIRS[@]}"; do [ -f "$d/$1" ] && { echo "$d/$1"; return 0; }; done; return 1; }
sha_of() { sha256sum "$1" | awk '{print $1}'; }
size_gb() { awk -v b="$(stat -c %s "$1" 2>/dev/null || stat -f %z "$1")" 'BEGIN{printf "%.1f", b/1000000000}'; }

if [ -n "$VERIFY" ]; then
  # every checksum in the README must be the checksum of a file that is here, and a line that names a file and carries a checksum must carry that file's checksum
  hashes=""; bad=0; seen=0
  for d in "${DIRS[@]}"; do for f in "$d"/*; do [ -f "$f" ] && grep -qF "$(basename "$f")" "$VERIFY" && hashes="$hashes $(sha_of "$f")"; done; done   # only files the README names are hashed (a folder may hold big unrelated files)
  readme_hashes="$(grep -Eo '[0-9a-f]{64}' "$VERIFY" | sort -u || true)"
  for h in $readme_hashes; do
    seen=$((seen + 1))
    case " $hashes " in *" $h "*) ;; *) echo "STALE: the README lists $h, which is the checksum of no file in: ${DIRS[*]}" >&2; bad=$((bad + 1)) ;; esac
  done
  for d in "${DIRS[@]}"; do for f in "$d"/*; do
    [ -f "$f" ] || continue
    grep -qF "$(basename "$f")" "$VERIFY" || continue
    # a line that names this file AND carries a checksum must carry this file's checksum (a sentence that merely mentions the file is fine)
    line_hash="$(grep -F "$(basename "$f")" "$VERIFY" | grep -Eo '[0-9a-f]{64}' | head -1 || true)"
    if [ -n "$line_hash" ] && [ "$line_hash" != "$(sha_of "$f")" ]; then
      echo "STALE: $(basename "$f") is $(sha_of "$f"), the README says $line_hash" >&2; bad=$((bad + 1))
    fi
  done; done
  echo "$seen checksum(s) in the README checked, $bad stale"
  [ "$bad" = 0 ] && [ "$seen" -gt 0 ]
  exit
fi

[ -n "$APPIMAGE" ] || usage
TDIR="$(cd "$(dirname "$TEMPLATE")" && pwd)"
# the app version has one source (the same file Tauri and the shell use)
APP_VERSION="$(grep -m1 '^version = ' "$TDIR/../src-tauri/Cargo.toml" | cut -d'"' -f2)"
[ -n "$APP_VERSION" ] || { echo "make-tester-readme: no version in $TDIR/../src-tauri/Cargo.toml" >&2; exit 1; }
TMP="$(mktemp)"; trap 'rm -f "$TMP" "$TMP.2"' EXIT
# includes first (one level), then the placeholders
while IFS= read -r line || [ -n "$line" ]; do
  if [[ "$line" =~ ^@INCLUDE:([^@]+)@$ ]]; then tr -d '\r' < "$TDIR/${BASH_REMATCH[1]}"; else printf '%s\n' "$line"; fi
done < <(tr -d '\r' < "$TEMPLATE") | sed "s|@APPIMAGE@|$APPIMAGE|g; s|@APP_VERSION@|$APP_VERSION|g" > "$TMP"

# @SHA256:name@ and @SIZE_GB:name@
while IFS= read -r token; do
  kind="${token#@}"; kind="${kind%%:*}"; name="${token#@*:}"; name="${name%@}"
  f="$(find_file "$name")" || { echo "make-tester-readme: no file named $name in: ${DIRS[*]} (a README must not carry a hash it did not compute)" >&2; exit 1; }
  case "$kind" in
    SHA256) value="$(sha_of "$f")" ;;
    SIZE_GB) value="$(size_gb "$f")" ;;
    *) echo "make-tester-readme: unknown placeholder $token" >&2; exit 1 ;;
  esac
  # replace every occurrence of this exact token (plain text, no pattern characters)
  text="$(cat "$TMP"; printf x)"; text="${text%x}"
  printf '%s' "${text//"$token"/$value}" > "$TMP"
done < <(grep -Eo '@(SHA256|SIZE_GB):[^@ ]+@' "$TMP" | sort -u)

if grep -Eq '@[A-Z0-9_]+[:@]' "$TMP"; then echo "make-tester-readme: placeholders left over:" >&2; grep -En '@[A-Z0-9_]+[:@]' "$TMP" >&2; exit 1; fi
cp "$TMP" "$OUT"
echo "wrote $OUT"
