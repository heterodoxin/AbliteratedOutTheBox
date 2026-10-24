# Strata container image, built from the engine COMPONENT PACKAGE: nothing is compiled on the user's side.
#
#   one image = the Terminal (bundled Python 3.12 + strata_cli / strata_control / serve, built exactly as the release archive)
#             + one engine package (CUDA 13, CUDA 12, HIP gfx120X, HIP gfx110X or SYCL) + the entrypoint
#
# Build it with docker/build.sh (it assembles the build context: the Terminal sources and the package folder):
#
#     docker/build.sh --flavor cuda --engine /path/to/0.1.45.3-linux-x86_64-cuda-cuda13 --tag strata:0.2.3-cuda13
#
# Plain `docker build` also works when the context folder has this layout (build.sh makes it):
#     terminal-src/   the files package_terminal.py needs (packaging/, strata_cli/, strata_control/, serve/, tools/, data/, ...)
#     engine/         one folder per engine package, exactly as `package_engine.py` wrote it (engine-component.json inside)
#     docker/         entrypoint.sh, strata-wrapper.sh, healthcheck.sh
#   docker build --build-arg FLAVOR=cuda -t strata:0.2.3-cuda13 -f Dockerfile <context>
#
# FLAVOR picks the small set of system libraries the runtime image adds and its base:
#   cuda  NVIDIA: the engine package carries cudart/cublas; the driver's libcuda comes from nvidia-container-toolkit
#   hip   AMD:    the engine package carries the ROCm runtime and kernels; the host's amdgpu driver and /dev/kfd are used
#   sycl  Intel:  the engine package carries oneAPI's SYCL runtime; this image adds the Level Zero loader and Intel's compute runtime
#         (Ubuntu 26.04 base: its compute runtime knows Arc A- and B-series; the 24.04 one does not know Battlemage)
#
# Only the legacy-builder-safe Dockerfile features are used (no heredocs, no --mount), so it builds with `docker build` as shipped
# in docker.io as well as with BuildKit.

ARG BASE_IMAGE=ubuntu:24.04

# ---- 1. the Terminal: the release archive, built here (pinned CPython 3.12 + hash-locked wheels, installed at build time only)
FROM ${BASE_IMAGE} AS terminal
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY terminal-src/ /src/
# the sources have no .git here (and the image has no git): build.sh passes the commit they are from, package_terminal.py writes it to COMMIT
ARG STRATA_COMMIT=
ENV STRATA_BUILD_COMMIT=${STRATA_COMMIT}
RUN python3 /src/packaging/package_terminal.py --out /out --cache /cache \
 && mkdir -p /opt/strata \
 && tar -xzf /out/strata-terminal-*-linux-x86_64.tar.gz -C /opt/strata \
 && mv /opt/strata/strata-terminal-* /opt/strata/terminal \
 && /opt/strata/terminal/bin/strata --version
# llama.cpp at the commit setup.py pins: fetched here, so the first start does not depend on GitHub. Only gguf-py (the model preparation
# tools) and the file setup.py looks for (ggml/CMakeLists.txt) are kept: 170 MB become a few
RUN cd /opt/strata/terminal && STRATA_CONTROL_INSTALL=1 STRATA_INSTALL_DIR=/opt/strata/seed PYTHONNOUSERSITE=1 \
      /opt/strata/terminal/python/bin/python3 -c "import setup; print(setup.get_llama_cpp())" \
 && rm -rf /opt/strata/seed/third_party/*.zip* \
 && cd /opt/strata/seed/third_party/llama.cpp \
 && find . -mindepth 1 -maxdepth 1 ! -name gguf-py ! -name ggml ! -name 'LICENSE*' -exec rm -rf {} + \
 && find ggml -mindepth 1 -depth ! -type d ! -name CMakeLists.txt -delete && find ggml -mindepth 1 -type d -empty -delete \
 && du -sh /opt/strata/seed

# ---- 2. the engine package, checked before it is allowed into the image: every file's hash, and every library it imports is
#         bundled or belongs to the OS / GPU driver (packaging/binary_deps.py, the check the packager runs)
FROM ${BASE_IMAGE} AS verify
COPY --from=terminal /opt/strata/terminal /opt/strata/terminal
COPY terminal-src/packaging/package_engine.py terminal-src/packaging/binary_deps.py /opt/strata/terminal/packaging/
COPY engine/ /engine/
RUN set -e; n=0; for d in /engine/*/; do \
      echo "== $d"; \
      /opt/strata/terminal/python/bin/python3 /opt/strata/terminal/packaging/package_engine.py --verify "$d"; \
      /opt/strata/terminal/python/bin/python3 /opt/strata/terminal/packaging/binary_deps.py "$d"; \
      n=$((n+1)); \
    done; [ "$n" -ge 1 ] || { echo "no engine package in the build context (engine/<package>/engine-component.json)"; exit 1; }

# ---- 3. the runtime image
FROM ${BASE_IMAGE}
ARG FLAVOR=cuda
ARG STRATA_VERSION=dev
ARG ENGINE_LABEL=
ENV DEBIAN_FRONTEND=noninteractive

# system libraries the engine packages leave to the OS (SYSTEM-LIBS.txt), plus tini (PID 1: forwards docker stop, reaps children)
RUN set -e; \
    pkgs="tini ca-certificates libgomp1 libatomic1 libnuma1 zlib1g"; \
    case "$FLAVOR" in \
      cuda) ;; \
      hip)  pkgs="$pkgs libdrm2 libdrm-amdgpu1 libelf1t64 libtinfo6 libpci3" ;; \
      sycl) pkgs="$pkgs libze1 libze-intel-gpu1 intel-opencl-icd" ;; \
      *) echo "FLAVOR must be cuda, hip or sycl"; exit 1 ;; \
    esac; \
    apt-get update; apt-get install -y --no-install-recommends $pkgs; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY --from=terminal /opt/strata/terminal /opt/strata/terminal
COPY --from=terminal /opt/strata/seed /opt/strata/seed
COPY --from=verify /engine/ /opt/strata/components/engine/
COPY docker/entrypoint.sh docker/healthcheck.sh /opt/strata/
COPY docker/strata-wrapper.sh /usr/local/bin/strata
RUN chmod 0755 /opt/strata/entrypoint.sh /opt/strata/healthcheck.sh /usr/local/bin/strata

# Does the dynamic linker find every library of the package inside this image? (the GPU driver's libcuda / libnvidia-* come from the host)
RUN set -e; \
    for d in /opt/strata/components/engine/*/; do \
      miss=$(find "$d" -type f \( -name 'strata*' -o -name '*.so' -o -name '*.so.*' \) ! -name '*.json' ! -name '*.txt' | while read -r f; do \
               head -c4 "$f" | grep -q ELF || continue; \
               LD_LIBRARY_PATH="$d" ldd "$f" 2>/dev/null | grep 'not found' | grep -v -E 'libcuda\.so|libnvidia|libnvcuvid|libze_intel_gpu|libhsa-runtime64|libdrm_amdgpu' | sed "s|^|$(basename "$f"): |"; \
             done | sort -u); \
      if [ -n "$miss" ]; then echo "MISSING LIBRARIES in $d"; echo "$miss"; exit 1; fi; \
      echo "libraries of $d resolve inside this image"; \
    done

# Everything that is yours lives in /data: model files, the prepared pack, presets, tuning results, the API key, logs.
# The engine and the Terminal are in the image; updating the image never touches /data.
ENV STRATA_COMPONENTS_DIR=/opt/strata/components \
    STRATA_INSTALL_DIR=/data/install \
    XDG_STATE_HOME=/data/state \
    XDG_CONFIG_HOME=/data/config \
    HOME=/data/home \
    PORT=8080 \
    LANG=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    STRATA_IN_DOCKER=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME ["/data"]
EXPOSE 8080

LABEL org.opencontainers.image.title="Strata" \
      org.opencontainers.image.source="https://github.com/Niko1221/Strata" \
      org.opencontainers.image.url="https://github.com/Niko1221/Strata" \
      org.opencontainers.image.documentation="https://github.com/Niko1221/Strata/blob/main/docs/DOCKER.md" \
      org.opencontainers.image.description="Strata local LLM server (Terminal + engine package), flavor ${FLAVOR}" \
      org.opencontainers.image.version="${STRATA_VERSION}" \
      org.opencontainers.image.licenses="see LICENSE" \
      com.strata.flavor="${FLAVOR}" \
      com.strata.engine="${ENGINE_LABEL}"

# The port only opens after the model is downloaded, prepared and loaded (minutes; hours for a first 70 GB download), so the
# start period is long: "starting" is not "unhealthy".
HEALTHCHECK --interval=30s --timeout=8s --start-period=2h --retries=3 CMD ["/opt/strata/healthcheck.sh"]

# `docker stop` sends SIGTERM to tini, which passes it to the entrypoint; the entrypoint runs `strata launch --stop` (drain, unload,
# free the GPU memory). Give it time: docker run --stop-timeout 120 (compose: stop_grace_period: 2m).
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strata/entrypoint.sh"]
CMD ["run"]
