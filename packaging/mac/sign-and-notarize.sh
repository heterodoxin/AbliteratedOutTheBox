#!/usr/bin/env bash
# Sign the Strata Mac app (dmg or .app) and, optionally, the Mac Terminal archive with the owner's Developer ID, then notarise
# them with Apple, so macOS opens them without the "damaged" / "unidentified developer" warning.
#
#   bash sign-and-notarize.sh <Strata-tester-mac.dmg | Strata.app> [--terminal <Strata-Terminal-Mac.zip>] [--updater-archive] [--out DIR] [--dry-run]
#   bash sign-and-notarize.sh --terminal <Strata-Terminal-Mac.zip>            # the Terminal archive alone
#
# This script runs on the OWNER'S Mac, where the certificate "Developer ID Application: Niko Veit (79LR4R7RSC)" is in the Keychain and the
# notarytool keychain profile "strata-notary" exists. It never asks for, reads or stores a password or a key: codesign and notarytool
# take them from the Keychain themselves. Nothing is uploaded except what notarytool sends to Apple's notary service.
#
# Environment overrides (the defaults are the owner's values):
#   SIGN_ID          the signing identity            (default: Developer ID Application: Niko Veit (79LR4R7RSC))
#   NOTARY_PROFILE   the notarytool keychain profile (default: strata-notary)
#   TMPDIR           where the work folder is made   (needs about 4 x the size of the input free)
#
# --dry-run: no Apple account is touched. Signs with the ad-hoc identity (SIGN_ID=-), skips notarytool/stapler/spctl-accept, and is
#   how this script is tested on a Mac that has no certificate. An ad-hoc signature has no Team ID, so in a dry run the hardened-runtime
#   "library validation" cannot be satisfied by libraries signed with the same identity; the dry run therefore adds
#   com.apple.security.cs.disable-library-validation to node and python ONLY so that the smoke tests can run. A real run never does.
#
# What it does, in order (each step prints OK or FAIL; the first FAIL stops the run and the summary says which):
#   1. copies the .app out of the dmg into a temporary folder;
#   2. signs every nested Mach-O inside-out (deepest first): hardened runtime, secure timestamp, entitlements only where needed;
#   3. rewrites the SHA-256 and size of every engine file it signed in the engine package's engine-component.json (Control refuses an
#      engine whose files do not match that manifest), then signs the .app itself;
#   4. codesign --verify --deep --strict, spctl -a -vvv;
#   5. smoke tests of the SIGNED copy (the app's own --self-test, node loading every native addon, python imports, the engine's --version,
#      and Control's own engine manifest check), so a signing mistake shows up here and not on a tester's Mac;
#   6. notarises the .app, staples it, rebuilds the dmg (same layout as apps/desktop/scripts/make-dmg.sh), signs the dmg, notarises the dmg,
#      staples it, spctl -a -t open --context context:primary-signature -vv;
#   7. Terminal archive: signs every Mach-O inside, re-zips, notarises the zip (a zip cannot be stapled: see the note at the end).
#   8. --updater-archive (opt-in, needs the app input): packs the signed, notarised, stapled .app into Strata.app.tar.gz, the file Tauri's in-app
#      updater downloads on a Mac (Strata.app at the root of the archive), and checks it by unpacking it again. The archive is NOT signed with the
#      Tauri updater key here: this script never touches that key. Sign it on the PC (npx tauri signer sign) and put it in the update file with
#      packaging/feed_tool.py set-app-platform (see packaging/UPDATES.md). Any Strata.app.tar.gz / .sig that `tauri build` made earlier holds the
#      UNSIGNED app and must not be published.
set -u
set -o pipefail

SIGN_ID="${SIGN_ID:-Developer ID Application: Niko Veit (79LR4R7RSC)}"
NOTARY_PROFILE="${NOTARY_PROFILE:-strata-notary}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENT_NODE_SRC="$HERE/entitlements-node.plist"

INPUT=""; TERMINAL=""; OUT=""; DRY=0; KEEP=0; SIGN_NESTED=0; UPDATER_ARCHIVE=0
WORK=""
LOG=""
declare -a RESULTS=()
declare -a MOUNTS=()
declare -a FINAL=()
NOTARY_ID=""

usage() {
  sed -n '2,/^set -u$/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --terminal)    [ $# -ge 2 ] || usage; TERMINAL="$2"; shift 2 ;;
    --out)         [ $# -ge 2 ] || usage; OUT="$2"; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    --keep-work)   KEEP=1; shift ;;
    --sign-nested) SIGN_NESTED=1; shift ;;
    --updater-archive) UPDATER_ARCHIVE=1; shift ;;
    -h|--help)     usage 0 ;;
    -*)            echo "unknown option: $1" >&2; usage ;;
    *)             if [ -z "$INPUT" ]; then INPUT="$1"; shift; else echo "unexpected argument: $1" >&2; usage; fi ;;
  esac
done
[ -n "$INPUT" ] || [ -n "$TERMINAL" ] || usage
if [ "$UPDATER_ARCHIVE" -eq 1 ] && [ -z "$INPUT" ]; then echo "--updater-archive needs the app (a .dmg or .app) as the input" >&2; usage; fi

# ----------------------------------------------------------------------------------------------------------------- output helpers
say() { printf '%s\n' "$*"; [ -z "$LOG" ] || printf '%s\n' "$*" >>"$LOG"; }
rec() { RESULTS+=("$1|$2|${3:-}"); say "  [$1] $2${3:+: $3}"; }

summary() {
  local rc=$1 line st name detail f sum
  say ""
  say "================================ SUMMARY ================================"
  if [ "${#RESULTS[@]}" -gt 0 ]; then
    for line in "${RESULTS[@]}"; do
      st="${line%%|*}"; line="${line#*|}"; name="${line%%|*}"; detail="${line#*|}"
      printf '  %-5s %s%s\n' "$st" "$name" "${detail:+  ($detail)}"
    done
  fi
  if [ "${#FINAL[@]}" -gt 0 ]; then
    say ""
    say "Final files:"
    for f in "${FINAL[@]}"; do
      sum="$(shasum -a 256 "$f" | awk '{print $1}')"
      say "  $f"
      say "    sha256 $sum"
      [ -z "$OUT" ] || printf '%s  %s\n' "$sum" "$(basename "$f")" >>"$OUT/SHA256SUMS.txt"
    done
  fi
  say ""
  if [ "$rc" -eq 0 ]; then
    if [ "$DRY" -eq 1 ]; then say "RESULT: DRY RUN finished. Nothing was sent to Apple; these files are ad-hoc signed and are NOT for release."
    else say "RESULT: all steps OK."; fi
  else
    say "RESULT: FAILED. Read the FAIL line above. Full log: ${LOG:-none}"
  fi
  if [ -n "$TERMINAL" ]; then
    say ""
    say "Note on the Terminal zip: Apple cannot staple a ticket to bare command-line programs in a zip, so the zip carries no ticket. Notarisation is"
    say "still recorded at Apple by the programs' hashes, and macOS checks that online the first time a quarantined (browser-downloaded) copy is opened."
    say "Offline, the first open of such a copy needs a connection. Installs done with 'curl | sh' are not quarantined at all, so this mainly helps zips"
    say "downloaded in a browser. The zip's sha256 changed: any feed or checksum list that names the old zip must be rebuilt from this one."
  fi
}

finish() { summary "$1"; exit "$1"; }
fail() { rec FAIL "$1" "${2:-}"; finish 1; }

# shellcheck disable=SC2329  # run by the EXIT trap below
cleanup() {
  local m
  if [ "${#MOUNTS[@]}" -gt 0 ]; then
    for m in "${MOUNTS[@]}"; do hdiutil detach "$m" -force >/dev/null 2>&1 || true; done
  fi
  if [ -n "$WORK" ] && [ -d "$WORK" ]; then
    if [ "$KEEP" -eq 1 ]; then echo "work folder kept: $WORK" >&2; else rm -rf "$WORK"; fi
  fi
}
trap cleanup EXIT
trap 'echo "interrupted" >&2; exit 130' INT TERM HUP

# run a command, appending all its output to the log; on failure show the last lines of it
logged() {
  local tmp rc
  tmp="$(mktemp "${WORK:-${TMPBASE:-/tmp}}/cmd.XXXXXX")"
  "$@" >"$tmp" 2>&1; rc=$?
  cat "$tmp" >>"${LOG:-/dev/null}"
  [ "$rc" -eq 0 ] || { tail -n 12 "$tmp" | sed 's/^/        /' >&2; }
  rm -f "$tmp"
  return "$rc"
}

# run a command with a time limit (macOS has no timeout(1)); 124 = timed out
run_timeout() {
  local secs=$1 i=0 pid
  shift
  "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$i" -ge $((secs * 2)) ]; then
      pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; sleep 1; kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 0.5; i=$((i + 1))
  done
  wait "$pid"
}

# ----------------------------------------------------------------------------------------------------------------- preflight
[ "$(uname -s)" = Darwin ] || { echo "This script runs on a Mac (it needs codesign, hdiutil, notarytool)." >&2; exit 2; }
for tool in codesign hdiutil ditto shasum file xattr perl PlistBuddy; do
  if [ "$tool" = PlistBuddy ]; then [ -x /usr/libexec/PlistBuddy ] || { echo "missing /usr/libexec/PlistBuddy" >&2; exit 2; }
  else command -v "$tool" >/dev/null 2>&1 || { echo "missing tool: $tool" >&2; exit 2; }; fi
done

if [ "$DRY" -eq 1 ]; then
  SIGN_ID="-"
else
  command -v xcrun >/dev/null 2>&1 || { echo "missing xcrun (install the Xcode command line tools: xcode-select --install)" >&2; exit 2; }
fi
if [ "$DRY" -eq 1 ]; then :
elif [ "$SIGN_ID" = "-" ]; then
  echo "SIGN_ID=- is an ad-hoc signature, which Apple cannot notarise. Add --dry-run to test with it." >&2; exit 2
fi

[ -z "$INPUT" ] || [ -e "$INPUT" ] || { echo "not found: $INPUT" >&2; exit 2; }
[ -z "$TERMINAL" ] || [ -f "$TERMINAL" ] || { echo "not found: $TERMINAL" >&2; exit 2; }

if [ -z "$OUT" ]; then
  first="${INPUT:-$TERMINAL}"
  OUT="$(cd "$(dirname "$first")" && pwd)/signed"
fi
mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 2; }
OUT="$(cd "$OUT" && pwd)"
TMPBASE="${TMPDIR:-/tmp}"; TMPBASE="${TMPBASE%/}"   # macOS TMPDIR ends in "/": a "//" in the path broke the app self-test
WORK="$(mktemp -d "$TMPBASE/strata-sign.XXXXXX")" || { echo "cannot make a work folder" >&2; exit 2; }
LOG="$OUT/sign-and-notarize.log"
: >"$LOG"
: >"$OUT/SHA256SUMS.txt"
say "Strata sign-and-notarize, $(date '+%Y-%m-%d %H:%M:%S')"
say "  identity : $SIGN_ID"
if [ "$DRY" -eq 1 ]; then say "  mode     : DRY RUN (ad-hoc, no notarisation)"; else say "  profile  : $NOTARY_PROFILE"; fi
say "  work     : $WORK"
say "  output   : $OUT"
say ""

need_kb=0
for f in "$INPUT" "$TERMINAL"; do
  [ -n "$f" ] || continue
  kb="$(du -sk "$f" | awk '{print $1}')"; need_kb=$((need_kb + kb * 5))
done
free_kb="$(df -k "$WORK" | awk 'NR==2 {print $4}')"
if [ "$free_kb" -lt "$need_kb" ]; then
  fail "free space" "about $((need_kb / 1048576)) GB needed in $WORK, $((free_kb / 1048576)) GB free; set TMPDIR to a bigger disk"
fi

if [ "$DRY" -eq 0 ]; then
  if ! security find-identity -v -p codesigning 2>/dev/null | grep -qF "\"$SIGN_ID\""; then
    fail "signing identity" "\"$SIGN_ID\" is not in your Keychain (security find-identity -v -p codesigning lists what is)"
  fi
  if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
    fail "notary profile" "profile \"$NOTARY_PROFILE\" does not work (create it once: xcrun notarytool store-credentials $NOTARY_PROFILE ...), or there is no internet"
  fi
  rec OK "preflight" "identity and notary profile found"
else
  rec OK "preflight" "dry run, ad-hoc identity"
fi

# ----------------------------------------------------------------------------------------------------------------- entitlements
# Every entitlement is a hole in the hardened runtime, so there are as few as work (each one was tested; see entitlements-node.plist).
#   node:   com.apple.security.cs.allow-jit  (V8 writes machine code into MAP_JIT memory; without it node dies at start with
#           "Failed to reserve virtual memory for CodeRange"). Nothing else: WebAssembly, regexp JIT and the native addons run with just this.
#   python, the engine, the app shell, rg, spawn-helper, every .dylib/.so/.node: none. (Python's ctypes callbacks and numpy run without any;
#   Metal compiles the engine's kernel sources through the system compiler service, which needs nothing from the engine process.)
#   disable-library-validation: NOT used. It would only be needed if a hardened program loaded a library signed by somebody else; every
#   library in these bundles is re-signed here with the one identity, so they share a Team ID. (Dry run only: see the header.)
ENT_NODE="$ENT_NODE_SRC"
ENT_DLV=""
if [ "$DRY" -eq 1 ]; then
  ENT_DLV="$WORK/dlv.plist"
  ENT_NODE="$WORK/node-dryrun.plist"
  cp "$ENT_NODE_SRC" "$ENT_NODE"
  /usr/libexec/PlistBuddy -c "Add :com.apple.security.cs.disable-library-validation bool true" "$ENT_NODE" >/dev/null
  printf '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict><key>com.apple.security.cs.disable-library-validation</key><true/></dict></plist>\n' >"$ENT_DLV"
fi
[ -f "$ENT_NODE_SRC" ] || fail "entitlements" "missing $ENT_NODE_SRC (keep it next to this script)"

ent_for() {
  case "$(basename "$1")" in
    node) printf '%s' "$ENT_NODE" ;;
    python|python3|python3.[0-9]*) printf '%s' "$ENT_DLV" ;;   # empty in a real run
    *) printf '' ;;
  esac
}

# ----------------------------------------------------------------------------------------------------------------- signing helpers
# Every Mach-O (programs, .dylib, .so, .node, universal binaries), deepest path first. Object files cannot be signed and are skipped.
find_macho() {
  find "$1" -type f -print0 2>/dev/null | xargs -0 file --mime-type 2>/dev/null \
    | sed -n 's/:[[:space:]]*application\/x-mach-binary$//p' | sed 's/ (for architecture [^)]*)$//' | sort -u \
    | while IFS= read -r f; do
        case "$(file -b "$f")" in *object*|*"core file"*) continue ;; esac
        printf '%s\n' "$f"
      done \
    | awk '{ n = gsub("/", "/"); print n "\t" $0 }' | sort -rn | cut -f2-
}

codesign_item() {   # path [extra codesign args...]
  local f=$1 ent try=1 rc=1
  shift
  local -a args=(--force --sign "$SIGN_ID" --options runtime)
  if [ "$SIGN_ID" = "-" ]; then args+=(--timestamp=none); else args+=(--timestamp); fi
  ent="$(ent_for "$f")"
  [ -z "$ent" ] || args+=(--entitlements "$ent")
  while [ "$try" -le 3 ]; do
    logged codesign "${args[@]}" "$@" "$f" && { rc=0; break; }
    say "        (attempt $try of 3 failed; the timestamp server is sometimes slow)"
    try=$((try + 1)); sleep 3
  done
  return "$rc"
}

codesign_plain() {   # a disk image: signature and secure timestamp, no hardened-runtime flag (that flag is for programs)
  local f=$1 try=1
  local -a args=(--force --sign "$SIGN_ID")
  if [ "$SIGN_ID" = "-" ]; then args+=(--timestamp=none); else args+=(--timestamp); fi
  while [ "$try" -le 3 ]; do
    logged codesign "${args[@]}" "$f" && return 0
    try=$((try + 1)); sleep 3
  done
  return 1
}

sign_tree() {   # folder label: all Mach-O inside-out, then nested bundles
  local dir=$1 label=$2 f n=0 bad=0 d
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if codesign_item "$f"; then n=$((n + 1)); else say "        could not sign: $f"; bad=$((bad + 1)); fi
  done < <(find_macho "$dir")
  [ "$bad" -eq 0 ] || fail "sign $label" "$bad file(s) could not be signed (see the log)"
  # nested bundles (frameworks, helper apps, XPC services), deepest first; there are none in today's app
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    [ "$d" != "$dir" ] || continue
    if [ -f "$d/Contents/Info.plist" ] || [ -f "$d/Resources/Info.plist" ] || [ -d "$d/Versions" ]; then
      codesign_item "$d" || fail "sign $label" "nested bundle $d"
    fi
  done < <(find "$dir" -type d \( -name '*.app' -o -name '*.framework' -o -name '*.xpc' -o -name '*.appex' \) 2>/dev/null | awk '{ n = gsub("/", "/"); print n "\t" $0 }' | sort -rn | cut -f2-)
  rec OK "sign $label" "$n Mach-O file(s) signed"
}

# The engine packages inside the bundle carry engine-component.json with the SHA-256 and size of every file; Control checks them before it
# starts an engine. Signing changes the engine binary, so the manifest must be brought up to date, or the signed app would refuse its own engine.
update_engine_manifests() {   # folder label
  local dir=$1 label=$2 m out rc n=0
  while IFS= read -r m; do
    [ -n "$m" ] || continue
    out="$(perl -MJSON::PP -MDigest::SHA -e '
      my ($file) = @ARGV; local $/; open my $fh, "<", $file or die "cannot read $file\n"; my $raw = <$fh>; close $fh;
      $raw =~ s/^\xEF\xBB\xBF//;
      my $json = JSON::PP->new->utf8->canonical->pretty; my $d = $json->decode($raw);
      (my $dir = $file) =~ s{/[^/]+$}{};
      if (defined $d->{signature}) { print "signed-manifest\n"; exit 3; }
      my $changed = 0;
      for my $e (@{ $d->{files} || [] }) {
        my $p = "$dir/$e->{path}"; next unless -f $p;
        open my $f, "<:raw", $p or die "cannot read $p\n"; my $sha = Digest::SHA->new(256); $sha->addfile($f); close $f;
        my $hex = $sha->hexdigest; my $size = -s $p;
        if (lc($e->{sha256}) ne $hex or $e->{bytes} != $size) { $e->{sha256} = $hex; $e->{bytes} = $size + 0; $changed++; }
      }
      if ($changed) { open my $o, ">", $file or die "cannot write $file\n"; print $o $json->encode($d); close $o; }
      print "$changed\n";
    ' "$m" 2>&1)"; rc=$?
    if [ "$rc" -eq 3 ]; then
      fail "engine manifest $label" "$m carries a release signature; changing a file would break it. Sign this engine package at its source, or re-sign its manifest with the release key"
    elif [ "$rc" -ne 0 ]; then
      fail "engine manifest $label" "$m: $out"
    fi
    n=$((n + 1)); say "        $m: $out entry(ies) updated"
  done < <(find "$dir" -name engine-component.json -type f 2>/dev/null)
  rec OK "engine manifests $label" "$n manifest(s) checked"
}

verify_signed() {   # path label
  local p=$1 label=$2
  if logged codesign --verify --deep --strict --verbose=2 "$p"; then rec OK "codesign --verify --deep --strict $label"
  else fail "codesign --verify --deep --strict $label" "see the log"; fi
}

# notarytool submit --wait; sets NOTARY_ID. On anything but Accepted, fetches Apple's log and fails.
notarize() {   # file label
  local f=$1 label=$2 out status
  out="$WORK/notary-$label.txt"
  say "        sending $(basename "$f") to Apple ($(du -h "$f" | awk '{print $1}')); this waits for the result, usually 2 to 15 minutes ..."
  xcrun notarytool submit "$f" --keychain-profile "$NOTARY_PROFILE" --wait --timeout 60m >"$out" 2>&1
  cat "$out" >>"$LOG"
  NOTARY_ID="$(awk '/^ *id:/ {print $2; exit}' "$out")"
  status="$(awk '/^ *status:/ {s = $2} END {print s}' "$out")"
  if [ "$status" = "Accepted" ]; then
    rec OK "notarise $label" "Accepted, submission $NOTARY_ID"
    return 0
  fi
  if [ -n "$NOTARY_ID" ]; then
    xcrun notarytool log --keychain-profile "$NOTARY_PROFILE" "$NOTARY_ID" "$OUT/notary-log-$label.json" >>"$LOG" 2>&1
    say ""
    say "---- Apple's notarisation log for $label (submission $NOTARY_ID), also saved to $OUT/notary-log-$label.json ----"
    cat "$OUT/notary-log-$label.json" 2>/dev/null | tee -a "$LOG"
    say "-------------------------------------------------------------------------------"
  else
    tail -n 15 "$out" | sed 's/^/        /'
  fi
  fail "notarise $label" "status: ${status:-no answer from notarytool}"
}

# a finished file goes to the output folder only when every step for it has passed, so a half-done file is never mistaken for a release
deliver() {   # file original
  local f=$1 orig=$2 dest
  dest="$OUT/$(basename "$f")"
  if [ -e "$dest" ] && [ "$dest" -ef "$orig" ]; then fail "write the result" "$dest is the input file itself; choose another --out folder"; fi
  cp -p "$f" "$dest" || fail "write the result" "cannot write $dest"
  FINAL+=("$dest")
}

zip_dir() {   # src_dir out.zip  (keeps symlinks and permissions, no AppleDouble clutter)
  ditto -c -k --norsrc --noextattr --noqtn --noacl --keepParent "$1" "$2"
}

# ----------------------------------------------------------------------------------------------------------------- smoke tests (signed copy)
smoke_app() {   # app
  local app=$1 res="$1/Contents/Resources" home="$WORK/smokehome" rep="$WORK/self-test.json" node py ctl rc f fails=0 out r engines=0
  mkdir -p "$home"
  node="$res/node/node"; ctl="$res/control"; py="$ctl/python/bin/python3"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1   # a smoke test must not write into the sealed bundle

  # 1. the app's own headless self-test: private node, host, UI, Control on the bundled python
  run_timeout 150 env HOME="$home" "$app/Contents/MacOS/strata-desktop" --self-test="$rep" >"$WORK/selftest.out" 2>&1; rc=$?
  cat "$WORK/selftest.out" >>"$LOG"
  if [ "$rc" -eq 0 ]; then rec OK "smoke: app --self-test (host, UI, Control)"
  else
    grep -E '"(name|ok|detail)"' "$rep" 2>/dev/null | paste - - - | grep 'false' | sed 's/^/        /'
    fail "smoke: app --self-test" "exit $rc; the signed app cannot start its host (see the log)"
  fi

  # 2. every native addon loads in the signed, hardened node (library validation, JIT)
  if [ -x "$node" ]; then
    out="$WORK/addons.txt"; : >"$out"
    while IFS= read -r f; do
      case "$(file -b "$f")" in *arm64*) ;; *) continue ;; esac
      r="$("$node" -e 'try { process.dlopen({ exports: {} }, process.argv[1]); console.log("OK"); } catch (e) { console.log("ERR " + String(e.message).split("\n")[0]); }' "$f" 2>&1 | head -n 1)"
      [ -n "$r" ] || r="ERR killed by the system (code signature)"
      printf '%s: %s\n' "$f" "$r" >>"$out"
    done < <(find "$res" -name '*.node' -type f 2>/dev/null)
    cat "$out" >>"$LOG"
    if grep -qiE 'code signature|team id|library validation|not valid for use|killed' "$out"; then
      grep -iE 'code signature|team id|library validation|not valid for use' "$out" | sed 's/^/        /'
      fail "smoke: node loads its native addons" "library validation refuses an addon (a library with a different signature)"
    fi
    if ! "$node" -e 'WebAssembly.instantiate(new Uint8Array([0,97,115,109,1,0,0,0])).then(()=>{let s=0;for(let i=0;i<2e6;i++)s+=i%7;if(!/a+b/.test("aab"))process.exit(3);console.log("jit ok")})' >>"$LOG" 2>&1; then
      fail "smoke: node JIT and WebAssembly" "the hardened node cannot run JIT code (entitlement missing?)"
    fi
    rec OK "smoke: node JIT, WebAssembly and native addons" "$(grep -c ': OK' "$out") addon(s) loaded"
  else
    rec SKIP "smoke: node" "no bundled node"
  fi

  # 3. Control's python imports its compiled modules
  if [ -x "$py" ]; then
    if (cd "$ctl" && env HOME="$home" "$py" -c '
import ctypes, importlib, sys
for m in ("numpy", "PIL.Image", "psutil", "yaml", "regex", "charset_normalizer", "markupsafe", "ssl", "sqlite3", "zlib", "lzma", "hashlib"):
    try: importlib.import_module(m)
    except ModuleNotFoundError: pass
cb = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(lambda a, b: 0)
arr = (ctypes.c_int * 4)(4, 3, 2, 1); ctypes.CDLL(None).qsort(arr, 4, 4, cb)
print("python ok", sys.version.split()[0])' >>"$LOG" 2>&1); then rec OK "smoke: bundled python imports and ctypes callbacks"
    else fail "smoke: bundled python" "an import or a ctypes callback failed under the hardened runtime (see the log)"; fi
  fi

  # 4. every bundled engine starts, and Control accepts its manifest (hashes of the signed files)
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    engines=$((engines + 1))
    if run_timeout 30 "$f" --version >"$WORK/engine-version.txt" 2>&1; then say "        $(head -n 1 "$WORK/engine-version.txt")"; else
      cat "$WORK/engine-version.txt" >>"$LOG"; fails=$((fails + 1)); say "        engine did not start: $f"
    fi
  done < <(find "$res/engine-bundle" -mindepth 2 -maxdepth 2 -type f \( -name strata -o -name strata-device -o -name strata-vision \) 2>/dev/null)
  [ "$fails" -eq 0 ] || fail "smoke: bundled engine starts" "$fails engine(s) did not start under the hardened runtime"
  if [ -x "$py" ] && [ -d "$res/engine-bundle" ]; then
    out="$(cd "$ctl" && env HOME="$home" "$py" -c '
import sys, pathlib
from strata_control import engine_component as ec
bad = 0; n = 0
for m in sorted(pathlib.Path(sys.argv[1]).glob("*/engine-component.json")):
    n += 1
    try:
        c = ec.read_manifest(m.parent); ec.verify(c, full=True); print("manifest ok", c.version, c.platform, c.backend)
    except Exception as e:
        bad += 1; print("MANIFEST PROBLEM", m, type(e).__name__, e)
sys.exit(1 if bad or not n else 0)' "$res/engine-bundle" 2>&1)"; rc=$?
    printf '%s\n' "$out" >>"$LOG"
    if [ "$rc" -eq 0 ]; then rec OK "smoke: engine manifest matches the signed files (Control's own check)"
    else printf '%s\n' "$out" | sed 's/^/        /'; fail "smoke: engine manifest" "Control would refuse the signed engine"; fi
  fi
  if [ "$engines" -gt 0 ]; then rec OK "smoke: bundled engine --version" "$engines engine(s), no model loaded"; else rec SKIP "smoke: bundled engine" "this app carries no engine"; fi
}

smoke_terminal() {   # extracted terminal folder
  local root=$1 py f strata home="$WORK/termhome" out
  mkdir -p "$home"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  py="$(find "$root" -maxdepth 4 -path '*/python/bin/python3' \( -type f -o -type l \) 2>/dev/null | head -n 1)"
  if [ -n "$py" ]; then
    if env HOME="$home" "$py" -c 'import ctypes, ssl, sqlite3, zlib; print("python ok")' >>"$LOG" 2>&1; then rec OK "smoke: terminal python starts"
    else fail "smoke: terminal python" "does not start under the hardened runtime"; fi
    strata="$(dirname "$py")/../../bin/strata"
    if [ -x "$strata" ]; then
      if out="$(env HOME="$home" STRATA_NO_ACQUIRE=1 "$strata" --version 2>&1)"; then rec OK "smoke: strata --version" "$(printf '%s' "$out" | head -n 1)"
      else printf '%s\n' "$out" | tail -n 5 | sed 's/^/        /'; fail "smoke: strata --version" "the Terminal launcher does not start"; fi
    fi
  else
    rec SKIP "smoke: terminal python" "no bundled python in this archive"
  fi
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    run_timeout 30 "$f" --version >>"$LOG" 2>&1 || fail "smoke: engine in the Terminal archive" "$f does not start"
  done < <(find "$root" -type f \( -name strata -o -name strata-device -o -name strata-vision \) -perm -u+x 2>/dev/null | while IFS= read -r c; do file "$c" | grep -q 'Mach-O' && printf '%s\n' "$c"; done)
}

# ----------------------------------------------------------------------------------------------------------------- the app
process_app() {
  local app_src="" appdir="$WORK/app" app version vol dmg_name zip out_dmg mnt bundle_real

  mkdir -p "$appdir"
  if [ -d "$INPUT" ] && { [[ "$INPUT" == *.app ]] || [[ "$INPUT" == *.app/ ]]; }; then
    app_src="${INPUT%/}"
    ditto --norsrc --noextattr --noqtn "$app_src" "$appdir/$(basename "$app_src")" || fail "copy the app" "ditto failed"
    app="$appdir/$(basename "$app_src")"
    dmg_name="Strata.dmg"
    rec OK "copy the app" "from $INPUT"
  elif [ -f "$INPUT" ] && [[ "$INPUT" == *.dmg ]]; then
    mnt="$WORK/mnt-in"; mkdir -p "$mnt"
    logged hdiutil attach -nobrowse -readonly -noautoopen -mountpoint "$mnt" "$INPUT" || fail "open the dmg" "hdiutil attach failed (is it already mounted?)"
    MOUNTS+=("$mnt")
    app_src="$(find "$mnt" -maxdepth 1 -name '*.app' -type d | head -n 1)"
    [ -n "$app_src" ] || fail "open the dmg" "no .app at the top of $INPUT"
    ditto --norsrc --noextattr --noqtn "$app_src" "$appdir/$(basename "$app_src")" || fail "copy the app" "ditto failed"
    hdiutil detach "$mnt" >/dev/null 2>&1 || hdiutil detach "$mnt" -force >/dev/null 2>&1
    app="$appdir/$(basename "$app_src")"
    dmg_name="$(basename "$INPUT")"
    rec OK "copy the app out of the dmg" "$(basename "$app") ($(du -sh "$app" | awk '{print $1}'))"
  else
    fail "input" "$INPUT is neither a .dmg nor a .app"
  fi
  [ -f "$app/Contents/Info.plist" ] || fail "input" "$app has no Contents/Info.plist"
  xattr -cr "$app" 2>/dev/null
  version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist" 2>/dev/null || echo "")"

  sign_tree "$app" "app contents"
  update_engine_manifests "$app" "app"
  if codesign_item "$app"; then rec OK "sign the .app itself"; else fail "sign the .app itself" "see the log"; fi

  verify_signed "$app" "app"
  if spctl -a -vvv -t exec "$app" >>"$LOG" 2>&1; then rec OK "spctl -a -vvv app" "accepted"
  elif [ "$DRY" -eq 1 ]; then rec INFO "spctl -a -vvv app" "rejected, as expected for an ad-hoc dry run"
  else rec INFO "spctl -a -vvv app" "rejected before notarisation (source=Unnotarized Developer ID); it is checked again after stapling"; fi

  smoke_app "$app"
  verify_signed "$app" "app after the smoke tests (nothing in the sealed bundle changed)"

  bundle_real="$app"
  if [ "$DRY" -eq 0 ]; then
    zip="$WORK/$(basename "$app" .app)-notarize.zip"
    zip_dir "$app" "$zip" || fail "zip the app for Apple" "ditto failed"
    notarize "$zip" "app"
    logged xcrun stapler staple "$app" || fail "staple the app" "stapler failed"
    logged xcrun stapler validate "$app" || fail "staple the app" "stapler validate failed"
    rec OK "staple the app"
    if spctl -a -vvv -t exec "$app" >>"$LOG" 2>&1; then rec OK "spctl -a -vvv app (after notarisation)" "accepted, Notarized Developer ID"
    else fail "spctl -a -vvv app (after notarisation)" "Gatekeeper still rejects the stapled app"; fi
    rm -f "$zip"
  fi

  # the dmg, same layout as apps/desktop/scripts/make-dmg.sh: the app, an Applications link, the volume icon, no Finder arrangement
  vol="Strata ${version:-app}"
  mkdir -p "$WORK/final"
  out_dmg="$WORK/final/$dmg_name"
  build_dmg "$bundle_real" "$out_dmg" "$vol" || fail "build the dmg" "hdiutil failed (see the log)"
  rec OK "build the dmg" "$vol"
  codesign_plain "$out_dmg" || fail "sign the dmg" "see the log"
  rec OK "sign the dmg"
  if [ "$DRY" -eq 0 ]; then
    notarize "$out_dmg" "dmg"
    logged xcrun stapler staple "$out_dmg" || fail "staple the dmg" "stapler failed"
    logged xcrun stapler validate "$out_dmg" || fail "staple the dmg" "stapler validate failed"
    rec OK "staple the dmg"
    if spctl -a -t open --context context:primary-signature -vv "$out_dmg" >>"$LOG" 2>&1; then rec OK "spctl -a -t open --context context:primary-signature -vv dmg" "accepted"
    else fail "spctl on the dmg" "Gatekeeper rejects the notarised dmg"; fi
  else
    rec INFO "notarise, staple and spctl on the dmg" "skipped in a dry run"
  fi
  logged hdiutil verify "$out_dmg" || fail "hdiutil verify" "the dmg is damaged"
  rec OK "hdiutil verify the dmg"

  # the app inside the finished dmg is still the signed one
  mnt="$WORK/mnt-out"; mkdir -p "$mnt"
  if logged hdiutil attach -nobrowse -readonly -noautoopen -mountpoint "$mnt" "$out_dmg"; then
    MOUNTS+=("$mnt")
    if logged codesign --verify --deep --strict "$mnt/$(basename "$app")"; then rec OK "the app inside the final dmg verifies"
    else hdiutil detach "$mnt" -force >/dev/null 2>&1; fail "the app inside the final dmg" "codesign --verify failed"; fi
    hdiutil detach "$mnt" >/dev/null 2>&1 || hdiutil detach "$mnt" -force >/dev/null 2>&1
  fi
  deliver "$out_dmg" "$INPUT"
  if [ "$UPDATER_ARCHIVE" -eq 1 ]; then make_updater_archive "$bundle_real"; fi
}

make_updater_archive() {   # the finished (signed, stapled) .app
  local app=$1 name chk top out
  name="$(basename "$app" .app).app.tar.gz"
  out="$WORK/final/$name"
  rm -f "$out"
  # the layout Tauri's bundler makes and its updater unpacks: the .app folder at the root of a gzip tar, nothing else
  # (COPYFILE_DISABLE keeps AppleDouble "._" files out; tar keeps symlinks and permissions)
  ( cd "$(dirname "$app")" && COPYFILE_DISABLE=1 tar -czf "$out" "$(basename "$app")" ) || fail "pack the updater archive" "tar failed"
  chk="$WORK/updater-check"; rm -rf "$chk"; mkdir -p "$chk"
  tar -xzf "$out" -C "$chk" || fail "check the updater archive" "tar cannot unpack it"
  top="$(ls -A "$chk")"
  [ "$top" = "$(basename "$app")" ] || fail "check the updater archive" "the archive root holds '$top', not just $(basename "$app")"
  find "$chk" -name '._*' | grep -q . && fail "check the updater archive" "AppleDouble files in the archive"
  if logged codesign --verify --deep --strict "$chk/$top"; then rec OK "updater archive: the unpacked app verifies"
  else fail "updater archive" "codesign --verify failed on the unpacked app"; fi
  if [ "$DRY" -eq 0 ]; then
    logged xcrun stapler validate "$chk/$top" || fail "updater archive" "the unpacked app has no valid notarisation ticket"
    if spctl -a -vvv -t exec "$chk/$top" >>"$LOG" 2>&1; then rec OK "updater archive: spctl accepts the unpacked app" "Notarized Developer ID"
    else fail "updater archive" "Gatekeeper rejects the unpacked app"; fi
  else
    rec INFO "updater archive: notarisation checks" "skipped in a dry run"
  fi
  rm -rf "$chk"
  rec OK "pack the updater archive" "$name"
  deliver "$out" "$INPUT"
}

build_dmg() {   # app out.dmg volname
  local app=$1 out=$2 vol=$3 rw="$WORK/rw.dmg" mnt="$WORK/mnt-build" kb mb n=0 icon
  kb="$(du -sk "$app" | awk '{print $1}')"; mb=$((kb * 13 / 10240 + 80))
  rm -f "$rw" "$out"
  logged hdiutil create -size "${mb}m" -fs HFS+ -volname "$vol" -type UDIF -ov "$rw" || return 1
  mkdir -p "$mnt"
  logged hdiutil attach -nobrowse -noautoopen -noverify -mountpoint "$mnt" "$rw" || return 1
  MOUNTS+=("$mnt")
  ditto --norsrc --noextattr --noqtn "$app" "$mnt/$(basename "$app")" || return 1
  ln -s /Applications "$mnt/Applications"
  icon="$app/Contents/Resources/Strata.icns"
  if [ -f "$icon" ] && command -v SetFile >/dev/null 2>&1; then cp "$icon" "$mnt/.VolumeIcon.icns" && SetFile -a C "$mnt" 2>/dev/null; fi
  sync
  until hdiutil detach "$mnt" >/dev/null 2>&1; do
    n=$((n + 1)); [ "$n" -lt 6 ] || { hdiutil detach "$mnt" -force >/dev/null 2>&1; break; }
    sleep 2
  done
  logged hdiutil convert "$rw" -format UDZO -imagekey zlib-level=9 -o "$out" || return 1
  rm -f "$rw"
}

# ----------------------------------------------------------------------------------------------------------------- the Terminal archive
repack_nested() {   # extracted root: sign inside .tar.gz/.tgz/.zip archives within the archive (opt-in)
  local root=$1 a tmp kind
  while IFS= read -r a; do
    [ -n "$a" ] || continue
    tmp="$WORK/nested.$$.$RANDOM"; mkdir -p "$tmp"
    case "$a" in
      *.zip) kind=zip; ditto -x -k "$a" "$tmp" ;;
      *)     kind=tar; tar -xzf "$a" -C "$tmp" ;;
    esac || { rm -rf "$tmp"; continue; }
    if [ -z "$(find_macho "$tmp" | head -n 1)" ]; then rm -rf "$tmp"; continue; fi
    sign_tree "$tmp" "nested $(basename "$a")"
    update_engine_manifests "$tmp" "nested $(basename "$a")"
    if [ "$kind" = zip ]; then
      ( cd "$tmp" && ditto -c -k --norsrc --noextattr --noqtn --noacl . "$a" )
    else
      ( cd "$tmp" && COPYFILE_DISABLE=1 tar -czf "$a" . )
    fi
    rm -rf "$tmp"
    say "        re-packed $(basename "$a"); a signed feed that pins its hash (strata-feed.json) must be regenerated and re-signed with the release key"
  done < <(find "$root" -type f \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \) 2>/dev/null)
}

process_terminal() {
  local src=$TERMINAL tdir="$WORK/terminal" top stem final notarize_zip name tops nested
  name="$(basename "$src")"
  mkdir -p "$tdir"
  case "$name" in
    *.zip)          ditto -x -k "$src" "$tdir" || fail "unpack the Terminal archive" "ditto failed" ;;
    *.tar.gz|*.tgz) tar -xzf "$src" -C "$tdir" || fail "unpack the Terminal archive" "tar failed" ;;
    *)              fail "unpack the Terminal archive" "$name is not a .zip or .tar.gz" ;;
  esac
  rm -rf "$tdir/__MACOSX"
  xattr -cr "$tdir" 2>/dev/null
  rec OK "unpack the Terminal archive" "$name"

  if [ -z "$(find_macho "$tdir" | head -n 1)" ]; then
    nested="$(find "$tdir" -type f \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \) 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$SIGN_NESTED" -eq 0 ] && [ "$nested" -gt 0 ]; then
      rec INFO "Terminal archive holds no program directly" "it contains $nested nested archive(s) (a feed folder); use --sign-nested to sign inside them"
    fi
  fi
  sign_tree "$tdir" "terminal"
  update_engine_manifests "$tdir" "terminal"
  [ "$SIGN_NESTED" -eq 0 ] || repack_nested "$tdir"
  if [ "$SIGN_NESTED" -eq 0 ]; then
    nested="$(find "$tdir" -type f \( -name '*.tar.gz' -o -name '*.tgz' \) 2>/dev/null | wc -l | tr -d ' ')"
    [ "$nested" -eq 0 ] || say "        $nested nested .tar.gz archive(s) were left untouched (they are installed by command-line tools and are not quarantined)"
  fi

  local bad=0 f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    codesign --verify --strict "$f" >>"$LOG" 2>&1 || { say "        bad signature: $f"; bad=$((bad + 1)); }
  done < <(find_macho "$tdir")
  [ "$bad" -eq 0 ] || fail "codesign --verify --strict (terminal)" "$bad file(s)"
  rec OK "codesign --verify --strict (terminal)" "every Mach-O"

  smoke_terminal "$tdir"

  # pack it again in the format it came in; Apple only takes a zip, so a tar.gz is notarised through a zip of the same files
  tops="$(find "$tdir" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
  stem="${name%.tar.gz}"; stem="${stem%.tgz}"; stem="${stem%.zip}"
  mkdir -p "$WORK/final"
  case "$name" in *.zip) final="$WORK/final/$stem.zip" ;; *) final="$WORK/final/$stem.tar.gz" ;; esac
  rm -f "$final"
  notarize_zip="$WORK/$stem-notarize.zip"; rm -f "$notarize_zip"
  if [ "$tops" -eq 1 ]; then
    top="$(find "$tdir" -mindepth 1 -maxdepth 1)"
    zip_dir "$top" "$notarize_zip" || fail "pack the Terminal archive" "ditto failed"
    case "$name" in *.zip) ;; *) ( cd "$tdir" && COPYFILE_DISABLE=1 tar -czf "$final" "$(basename "$top")" ) || fail "pack the Terminal archive" "tar failed" ;; esac
  else
    ( cd "$tdir" && ditto -c -k --norsrc --noextattr --noqtn --noacl . "$notarize_zip" ) || fail "pack the Terminal archive" "ditto failed"
    case "$name" in *.zip) ;; *) ( cd "$tdir" && COPYFILE_DISABLE=1 tar -czf "$final" . ) || fail "pack the Terminal archive" "tar failed" ;; esac
  fi
  case "$name" in *.zip) cp "$notarize_zip" "$final" ;; esac
  rec OK "pack the Terminal archive" "$(basename "$final")"

  if [ "$DRY" -eq 0 ]; then
    notarize "$notarize_zip" "terminal"
    rec INFO "staple the Terminal archive" "not possible for bare programs in a zip; Apple holds the ticket online"
  else
    rec INFO "notarise the Terminal archive" "skipped in a dry run"
  fi
  deliver "$final" "$src"
}

# ----------------------------------------------------------------------------------------------------------------- run
if [ -n "$INPUT" ]; then process_app; fi
if [ -n "$TERMINAL" ]; then process_terminal; fi
finish 0
