# Strata SYCL port - development image for the Intel Arc path.
# Base: the llama.cpp SYCL image already on the box (oneAPI 2026.1 compiler, MKL, Level Zero for Xe2).
# Adds SYCLomatic (dpct, "Intel DPC++ Compatibility Tool") for the CUDA -> SYCL migration, and the build tools.
FROM ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      intel-oneapi-dpcpp-ct ninja-build unzip python3-pip python3-venv rsync \
    && rm -rf /var/lib/apt/lists/*
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
ENTRYPOINT ["/bin/bash", "-lc"]
