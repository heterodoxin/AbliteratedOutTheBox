#!/bin/sh
# Build the Strata images of one release from its engine packages and, only with --push, push them to the registry.
# The GitHub Actions workflow (.github/workflows/docker-images.yml) runs this script once per flavor; you can run it by hand too.
#
#   docker/publish.sh --release-tag v0.2.3                        # builds every flavor the release has a package for; pushes nothing
#   docker/publish.sh --release-tag v0.2.3 --flavors cuda13,cuda12 --push
#   docker/publish.sh --source ~/release-folder --version 0.2.3-test --flavors cuda13 --no-latest
#
# For every flavor:
#   1. docker/fetch-engines.py downloads the engine package from the release (a GitHub release URL, or --source) and checks it
#      against the signed update feed: the feed's Ed25519 signature, the archive's size and SHA-256, the manifest hash, the package
#      signature and every file's hash. A package that fails stops the run before anything is built.
#   2. docker/build.sh builds <image>:<version>-<flavor> (the Dockerfile checks the package again, and that its libraries resolve).
#   3. a short check that the image starts and the Terminal inside it reports the version.
#   4. with --push: docker push of <image>:<version>-<flavor> and, unless --no-latest, <image>:latest-<flavor>. You must be logged in
#      to the registry already (`docker login ghcr.io`; the workflow logs in with GITHUB_TOKEN).
#
#   --release-tag TAG    the release, for example v0.2.3; the image version is the tag without the leading v
#   --source DIR|URL     use this folder or URL (with strata-feed.json, its .sig and the archives) instead of the GitHub release
#   --version V          image version (default: the tag without v)
#   --flavors LIST       comma separated (default: all that the release has: cuda13 cuda12 rocm-gfx120x rocm-gfx110x rocm-gfx1151 sycl)
#   --image NAME         image repository (default ghcr.io/niko1221/strata)
#   --work DIR           where packages are unpacked (default a temporary folder, removed at the end)
#   --push               push the images (default: build only)
#   --no-latest          do not tag / push latest-<flavor>
#   --keep-images        do not remove the local images after the run (default: removed only when this script pushed them)
# Environment: STRATA_DOCKER_BUILD_OPTS adds options to `docker build` (see build.sh).
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(CDPATH= cd -- "$here/.." && pwd -P)
tag=""; source=""; version=""; flavors=""; image="ghcr.io/niko1221/strata"; work=""; push=""; latest=1; keepimg=""
while [ $# -gt 0 ]; do
  case "$1" in
    --release-tag) tag=$2; shift 2 ;;
    --source) source=$2; shift 2 ;;
    --version) version=$2; shift 2 ;;
    --flavors) flavors=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --work) work=$2; shift 2 ;;
    --push) push=1; shift ;;
    --no-latest) latest=""; shift ;;
    --keep-images) keepimg=1; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "publish.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$tag" ] || [ -n "$source" ] || { echo "publish.sh: give --release-tag or --source" >&2; exit 2; }
if [ -z "$version" ]; then
  [ -n "$tag" ] || { echo "publish.sh: --version is required with --source" >&2; exit 2; }
  version=${tag#v}
fi
case "$image" in *[A-Z]*) echo "publish.sh: the image name must be lower case ($image)" >&2; exit 2 ;; esac
command -v docker >/dev/null || { echo "publish.sh: docker not found" >&2; exit 2; }
PY=${PYTHON:-python3}

own_work=""
if [ -z "$work" ]; then work=$(mktemp -d "${TMPDIR:-/tmp}/strata-docker-pkgs.XXXXXX"); own_work=1; fi
mkdir -p "$work"
cleanup() { if [ -n "$own_work" ]; then rm -rf "$work"; fi; }
trap cleanup EXIT

built=""
# one flavor at a time: the package is deleted after its image is built, so the disk never holds more than one
flavor_list=${flavors:-cuda13,cuda12,rocm-gfx120x,rocm-gfx110x,rocm-gfx1151,sycl}
for flavor in $(echo "$flavor_list" | tr ',' ' '); do
  echo "=== $flavor"
  src_arg="--release-tag $tag"; [ -n "$source" ] && src_arg="--source $source"
  # a failed fetch (bad signature, bad hash) must stop the run: it is not a skipped flavor
  $PY "$here/fetch-engines.py" --out "$work" --skip-missing --flavors "$flavor" $src_arg > "$work/.fetch.out"
  line=$(tail -1 "$work/.fetch.out")
  bflavor=$(echo "$line" | cut -f2); folder=$(echo "$line" | cut -f3)
  if [ "$folder" = "SKIP" ] || [ -z "$folder" ]; then echo "$flavor: no package in this release, skipped"; continue; fi
  img="$image:$version-$flavor"
  "$here/build.sh" --flavor "$bflavor" --engine "$folder" --tag "$img" --version "$version" --repo "$repo"
  [ -n "$latest" ] && docker tag "$img" "$image:latest-$flavor"
  # the image starts and its Terminal runs (no GPU needed for this)
  docker run --rm --entrypoint /opt/strata/terminal/bin/strata "$img" --version
  rm -rf "$folder"
  if [ -n "$push" ]; then
    docker push "$img"
    [ -n "$latest" ] && docker push "$image:latest-$flavor"
    echo "pushed $img$( [ -n "$latest" ] && echo " and $image:latest-$flavor")"
    docker image inspect "$img" --format '{{index .RepoDigests 0}}' 2>/dev/null || true
    [ -z "$keepimg" ] && { docker rmi "$img" >/dev/null 2>&1 || true; [ -n "$latest" ] && { docker rmi "$image:latest-$flavor" >/dev/null 2>&1 || true; }; }
  else
    echo "built $img (not pushed; add --push)"
  fi
  built="$built $flavor"
done
[ -n "$built" ] || { echo "publish.sh: no flavor was built (the release has none of: $flavor_list)" >&2; exit 3; }
echo "done:$built"
