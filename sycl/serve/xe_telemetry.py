"""Intel Arc (xe driver) readings for the server's Monitor tab, in the shape of serve/telemetry.py's readers.

Kept out of serve/telemetry.py so the SYCL port touches no shared file: sycl/serve/server_intel.py installs it as
telemetry.gpu_reader when there is no NVIDIA card.  Load and VRAM need root (xe's only VRAM accounting is per-client
fdinfo), so they come from a root sampler's /run/gpustat.json (sycl/tools/gpustat.py); temperature, power and the
PCIe link are read from sysfs, so they show without it.
"""
from __future__ import annotations

import os
import sys
import time


# the cards sycl/setup_intel.py names by PCI id (INTEL_ARC)
ARC_NAMES = {"e223": "Arc Pro B70", "e221": "Arc Pro B60", "e211": "Arc Pro B60", "e20b": "Arc B580", "e20c": "Arc B570",
             "e212": "Arc B50", "56a0": "Arc A770", "56a1": "Arc A750", "56a2": "Arc A580", "56a5": "Arc A380",
             "56a6": "Arc A310", "5690": "Arc A770M"}


class _XeGpu:
    """The Intel Arc readings in NVML's shape. Load and VRAM come from /run/gpustat.json (a root sampler: the only
    VRAM accounting xe exposes is per-client fdinfo, readable by root only); temperature, power and the PCIe link are
    read from sysfs too, so they show without the sampler."""
    STAT = "/run/gpustat.json"

    def __init__(self, index=0):
        import glob
        self.dev = None
        cards = []                              # #2001: the Intel cards in sysfs order, as setup numbers them
        for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            try:
                if open(f"{d}/vendor").read().strip() == "0x8086" and os.path.isdir(f"{d}/tile0"):
                    cards.append(os.path.realpath(d))
            except OSError:
                continue
        if 0 <= index < len(cards):
            self.dev = cards[index]
        self.hwmon = None
        for n in glob.glob(f"{self.dev}/hwmon/hwmon*/name") if self.dev else []:
            self.hwmon = os.path.dirname(n)
        self._e = None                          # (t, card energy uJ) for power without the sampler

    def ok(self):
        return self.dev is not None and sys.platform.startswith("linux")

    def name(self):
        st = self._stat()
        if (st or {}).get("name"):
            return st["name"]
        try:                                    # the PCI id's name (setup_intel.INTEL_ARC), not a generic one
            with open(f"{self.dev}/device") as f:
                return "Intel " + ARC_NAMES.get(f.read().strip().lower().replace("0x", ""), "Arc GPU")
        except (OSError, TypeError):
            return "Intel Arc GPU"

    def _stat(self):
        try:
            import json
            with open(self.STAT) as f:
                st = json.load(f)
            return st if time.time() - float(st.get("ts", 0)) < 15 else None
        except (OSError, ValueError):
            return None

    def _rd(self, p):
        try:
            return open(p).read().strip()
        except OSError:
            return None

    def _link(self):
        """The link the card trained at: its own functions sit behind an internal x1 switch, so the first port up
        the path wider than x1; the max is what card AND slot allow (the root port caps it)."""
        gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
        d, chain = self.dev, []
        while d and d.startswith("/sys/devices/pci") and os.path.exists(f"{d}/current_link_speed"):
            chain.append(d)
            d = os.path.dirname(d)
        for p in chain:
            w = int(self._rd(f"{p}/current_link_width") or 0)
            if w > 1:
                g = lambda q, k: gen.get((self._rd(f"{q}/{k}_link_speed") or "").split(" ")[0])
                card, slot = g(p, "max"), g(chain[-1], "max")
                return g(p, "current"), min(x for x in (card, slot) if x) if (card or slot) else None, w
        return None, None, None

    def read(self):
        out = {}
        st = self._stat()
        if st:
            out["util"] = st.get("busy_pct")
            if st.get("vram_used_mb") is not None:
                out["mem_used"] = st["vram_used_mb"] * 2**20
                out["mem_total"] = st.get("vram_total_mb", 0) * 2**20 or None
            out["temp"] = st.get("temp_pkg")
            out["power"] = st.get("power_w")
            out["power_limit"] = st.get("power_cap_w")
            out["vram_temp"] = st.get("temp_vram_max") or st.get("temp_vram")
        if self.hwmon:
            if out.get("temp") is None:
                for lab in os.listdir(self.hwmon):
                    if lab.endswith("_label") and self._rd(f"{self.hwmon}/{lab}") == "pkg":
                        v = self._rd(f"{self.hwmon}/{lab[:-6]}_input")
                        out["temp"] = int(v) / 1000 if v else None
            if out.get("power") is None:
                e, t = self._rd(f"{self.hwmon}/energy1_input"), time.time()
                if e and e.isdigit():
                    if self._e and t > self._e[0] and int(e) >= self._e[1]:
                        out["power"] = (int(e) - self._e[1]) / 1e6 / (t - self._e[0])
                    self._e = (t, int(e))
            if out.get("power_limit") is None:
                cap = self._rd(f"{self.hwmon}/power1_cap")
                out["power_limit"] = int(cap) / 1e6 if cap and cap.isdigit() and int(cap) > 0 else None
        out["pcie_gen"], out["pcie_gen_max"], out["pcie_width"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None    # xe exposes no PCIe traffic counters
        return out
