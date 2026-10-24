#!/bin/sh
# Build a Strata image from an engine component package. Run it on a Linux x86_64 machine with Docker (it needs the internet
# for the base image and for the Terminal's pinned Python and locked wheels; the engine package is local).
#
#   docker/build.sh --flavor cuda --engine ~/pkgs/0.1.45.3-linux-x86_64-cuda-cuda13 --tag strata:0.2.3-cuda13
#   docker/build.sh --flavor hip  --engine ~/pkgs/0.1.45.3-linux-x86_64-hip-gfx120X  --tag strata:0.2.3-rocm-gfx120x
#   docker/build.sh --flavor sycl --engine ~/pkgs/0.1.45.3-linux-x86_64-sycl         --tag strata:0.2.3-sycl
#
# --engine is the package FOLDER that packaging/package_engine.py wrote (engine-component.json inside). It can be given more than
# once to put several packages in one image (the Terminal picks the one that fits the card).
#
#   --flavor cuda|hip|sycl   which system libraries the runtime adds (see docker/Dockerfile)
#   --tag NAME               image name (default strata:<version>-<flavor>)
#   --base IMAGE             base image (default ubuntu:24.04; sycl: ubuntu:26.04)
#   --repo DIR               the Strata source tree (default: the one this script is in)
#   --version V              label (default: read from CMakeLists.txt)
#   --ctx DIR                keep the build context here (default: a temporary folder, removed after the build)
#   --no-cache               docker build --no-cache
#   --ctx-only               assemble the context and stop
# Environment: STRATA_DOCKER_BUILD_OPTS adds options to `docker build` (for example --network host).
# Nothing is pushed anywhere; the image stays in the local Docker.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(CDPATH= cd -- "$here/.." && pwd -P)
flavor=""; tag=""; base=""; version=""; ctx=""; nocache=""; ctxonly=""
engines=""
while [ $# -gt 0 ]; do
  case "$1" in
    --flavor) flavor=$2; shift 2 ;;
    --engine) engines="$engines
$2"; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --base) base=$2; shift 2 ;;
    --repo) repo=$(CDPATH= cd -- "$2" && pwd -P); shift 2 ;;
    --version) version=$2; shift 2 ;;
    --ctx) ctx=$2; shift 2 ;;
    --no-cache) nocache=--no-cache; shift ;;
    --ctx-only) ctxonly=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "build.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$flavor" ] || { echo "build.sh: --flavor cuda|hip|sycl is required" >&2; exit 2; }
case "$flavor" in cuda|hip|sycl) ;; *) echo "build.sh: --flavor must be cuda, hip or sycl" >&2; exit 2 ;; esac
[ -n "$engines" ] || { echo "build.sh: --engine <package folder> is required" >&2; exit 2; }
[ -n "$base" ] || { [ "$flavor" = sycl ] && base=ubuntu:26.04 || base=ubuntu:24.04; }
if [ -z "$version" ]; then
  version=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' "$repo/CMakeLists.txt" | head -1)
  version=${version:-dev}
fi

own_ctx=""
if [ -z "$ctx" ]; then ctx=$(mktemp -d "${TMPDIR:-/tmp}/strata-docker-ctx.XXXXXX"); own_ctx=1; fi
mkdir -p "$ctx/terminal-src" "$ctx/engine" "$ctx/docker"
cleanup() { if [ -n "$own_ctx" ] && [ -z "$ctxonly" ]; then rm -rf "$ctx"; fi; }
trap cleanup EXIT

# the commit the sources are from (the build context has no .git; package_terminal.py writes it into the archive's COMMIT)
commit=${STRATA_BUILD_COMMIT:-$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)}
case "$commit" in *[!0-9a-f]*|"") commit="" ;; esac

# 1. the Terminal's sources: exactly what packaging/package_terminal.py copies (the working tree, so local edits are in the image)
cd "$repo"
paths="packaging bin strata.py strata_control strata_cli serve requirements.txt requirements-monitor.txt docs/MANAGEMENT_PROTOCOL.md docs/TERMINAL_LAUNCH.md data bench/prompts/standard-v1 setup.py ref/load.py CMakeLists.txt"
if git rev-parse --git-dir >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  { git ls-files -z -- $paths ':(glob)tools/*.py'; } | tar --null -T - -cf - | tar -xf - -C "$ctx/terminal-src"
else
  # shellcheck disable=SC2086
  tar -cf - --exclude='__pycache__' $paths tools/*.py | tar -xf - -C "$ctx/terminal-src"
fi

# 2. the engine package(s): hard-linked when on the same file system (no second copy of GBs), copied otherwise
for e in $engines; do
  [ -f "$e/engine-component.json" ] || { echo "build.sh: $e has no engine-component.json (give the package folder)" >&2; exit 2; }
  dest="$ctx/engine/$(basename "$e")"
  rm -rf "$dest"
  cp -al "$e" "$dest" 2>/dev/null || { rm -rf "$dest"; cp -a "$e" "$dest"; }     # (a failed hard-link attempt leaves a half-made folder)
  # a package that went through a zip or a Windows share lost its executable bit: the manifest hashes content only, so restore it on
  # OUR copy (break the hard link first, so the original is not changed)
  entry=$(sed -n 's/.*"entrypoint": *"\([^"]*\)".*/\1/p' "$dest/engine-component.json" | head -1)
  if [ -n "$entry" ] && [ -f "$dest/$entry" ] && [ ! -x "$dest/$entry" ]; then
    echo "note: $entry in $e is not executable; making it executable in the image copy"
    cp --remove-destination "$e/$entry" "$dest/$entry" && chmod 0755 "$dest/$entry"
  fi
done

# 3. the Dockerfile and the container scripts
cp "$here/Dockerfile" "$ctx/Dockerfile"
cp "$here/entrypoint.sh" "$here/strata-wrapper.sh" "$here/healthcheck.sh" "$ctx/docker/"
tr -d '\r' < "$here/entrypoint.sh" > "$ctx/docker/entrypoint.sh"          # a checkout with CRLF line ends must not break the scripts
tr -d '\r' < "$here/strata-wrapper.sh" > "$ctx/docker/strata-wrapper.sh"
tr -d '\r' < "$here/healthcheck.sh" > "$ctx/docker/healthcheck.sh"

engine_label=$(for e in $engines; do basename "$e"; done | tr '\n' ' ' | sed 's/ $//')
[ -n "$tag" ] || tag="strata:$version-$flavor"
echo "context: $ctx"
echo "flavor $flavor, base $base, engine(s): $engine_label, tag $tag"
[ -n "$ctxonly" ] && { echo "context kept (--ctx-only)"; exit 0; }

# shellcheck disable=SC2086
docker build $nocache ${STRATA_DOCKER_BUILD_OPTS:-} \
  --build-arg "BASE_IMAGE=$base" --build-arg "FLAVOR=$flavor" \
  --build-arg "STRATA_COMMIT=$commit" --build-arg "STRATA_VERSION=$version" --build-arg "ENGINE_LABEL=$engine_label" \
  -t "$tag" "$ctx"
docker image inspect "$tag" --format 'built {{.RepoTags}} {{.Size}} bytes'
echo "Not pushed anywhere. Size on disk: $(docker image inspect "$tag" --format '{{.Size}}') bytes."
