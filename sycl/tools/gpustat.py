#!/usr/bin/env python3
"""GPU telemetry sampler for an Intel Arc on the xe driver (runs as root: sycl/tools/gpustat.service).
Writes /run/gpustat.json every INTERVAL s; Strata's web app (serve/telemetry.py) shows it in the Monitor tab:
  name, vram_used_mb / vram_total_mb (sum of drm-resident-vram0 over unique drm
  clients from /proc/*/fdinfo - the only VRAM accounting xe exposes; root-only),
  busy_pct (compute/render cycle deltas), temp_pkg/temp_vram (hwmon 'xe'),
  power_w / pkg_power_w (energy counter deltas over a POWER_WINDOW s window: the counters update in
  bursts, so a 3 s delta swings 0.6 <-> 48 W), power_cap_w, fan_rpm, freq_mhz, the PCIe link the card
  trained at (the first upstream port wider than x1 - the card's own functions sit behind an internal
  x1 switch), the hottest VRAM channel, and the host's 1-minute load. Device totals only.
Without it the Monitor tab still shows temperature, power and the PCIe link (sysfs), but not load or VRAM: xe
reports those per client in /proc/*/fdinfo, which only root can read.
Install:  sudo install -m 755 sycl/tools/gpustat.py /usr/local/sbin/
          sudo install -m 644 sycl/tools/gpustat.service /etc/systemd/system/ && sudo systemctl enable --now gpustat
Env: GPUSTAT_INTERVAL (s), GPUSTAT_POWER_WINDOW (s), GPUSTAT_PCI (the card, e.g. 0000:03:00.0; default the first
Intel GPU on xe), GPUSTAT_VRAM_MB (the total; default the card's largest PCI BAR, which maps all of VRAM with
resizable BAR - a little above what the runtime reports as usable).
"""
import collections, glob, json, os, subprocess, time

OUT = "/run/gpustat.json"
INTERVAL = float(os.environ.get("GPUSTAT_INTERVAL", "3"))   # each sample scans /proc/*/fd; keep it modest
NAMES = {"e223": "Intel(R) Arc(TM) Pro B70 Graphics"}


def find_card():
    """The first Intel GPU on the xe driver (it has tile*/gt*), as a PCI address."""
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        try:
            if open(f"{d}/vendor").read().strip() == "0x8086" and glob.glob(f"{d}/tile*/gt*"):
                return os.path.basename(os.path.realpath(d))
        except OSError:
            continue
    return None


PCI = os.environ.get("GPUSTAT_PCI") or find_card() or "0000:00:00.0"


def vram_total_mb():
    if os.environ.get("GPUSTAT_VRAM_MB"):
        return int(os.environ["GPUSTAT_VRAM_MB"])
    try:                                   # with resizable BAR the largest BAR maps all of VRAM
        sizes = [int(e, 16) - int(s, 16) + 1 for s, e, _ in
                 (l.split() for l in open(f"/sys/bus/pci/devices/{PCI}/resource") if l.strip())]
        return max(sizes) // 2**20
    except (OSError, ValueError):
        return 0


def card_name(dev_id):
    if dev_id in NAMES:
        return NAMES[dev_id]
    try:
        out = subprocess.run(["lspci", "-mm", "-s", PCI], capture_output=True, text=True, timeout=5).stdout
        parts = [p.strip('"') for p in out.split('" "')]
        if len(parts) >= 3 and parts[2]:
            return parts[2]
    except (OSError, subprocess.SubprocessError):
        pass
    return f"Intel GPU 8086:{dev_id}"


TOTAL_MB = vram_total_mb()
POWER_WINDOW = float(os.environ.get("GPUSTAT_POWER_WINDOW", "15"))
GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}

def pcie_link():
    """The link the card trained at: walk up from the GPU function to the first port wider than x1."""
    d = os.path.realpath(f"/sys/bus/pci/devices/{PCI}")
    chain = []
    while d.startswith("/sys/devices/pci") and os.path.exists(f"{d}/current_link_speed"):
        chain.append(d); d = os.path.dirname(d)
    def link(p, kind):
        sp = (rd(f"{p}/{kind}_link_speed", "") or "").split(" ")[0]
        return {"gen": GEN.get(sp), "gts": sp, "width": int(rd(f"{p}/{kind}_link_width", "0") or 0)}
    for i, p in enumerate(chain):
        cur = link(p, "current")
        if cur["width"] > 1:
            root = chain[-1]                                  # the CPU's root port: what the slot can do
            return {"cur": cur, "card_max": link(p, "max"), "slot_max": link(root, "max")}
    return None

def hwmon_dir():
    for n in glob.glob("/sys/class/hwmon/hwmon*/name"):
        if open(n).read().strip() == "xe":
            return os.path.dirname(n)
    return None

def rd(p, default=None):
    try:
        return open(p).read().strip()
    except Exception:
        return default

def hw_temps(h):
    t = {}
    for lab in glob.glob(f"{h}/temp*_label"):
        name = rd(lab); v = rd(lab.replace("_label", "_input"))
        if name and v: t[name] = int(v) / 1000
    return t

def clients():
    """{client_id: {pid, comm, vram_kb, cycles, total_cycles}} across all processes."""
    res = {}
    for pdir in glob.glob("/proc/[0-9]*"):
        try:
            fds = os.listdir(f"{pdir}/fd")
        except Exception:
            continue
        for fd in fds:
            try:
                if not os.readlink(f"{pdir}/fd/{fd}").startswith("/dev/dri/"):
                    continue
                info = dict(l.split(":\t", 1) for l in open(f"{pdir}/fdinfo/{fd}").read().splitlines() if l.startswith("drm-"))
            except Exception:
                continue
            cid = info.get("drm-client-id")
            if not cid or cid in res:
                continue
            kb = lambda k: int(info.get(k, "0").split()[0] or 0)
            cyc = max(kb("drm-cycles-rcs"), kb("drm-cycles-ccs"))
            tot = max(kb("drm-total-cycles-rcs"), kb("drm-total-cycles-ccs"))
            res[cid] = {"pid": int(pdir[6:]), "comm": rd(f"{pdir}/comm", "?"),
                        "cg": rd(f"{pdir}/cgroup", ""), "vram_kb": kb("drm-resident-vram0"), "cycles": cyc, "total": tot}
    return res

def main():
    h = hwmon_dir()
    dev_id = (rd(f"/sys/bus/pci/devices/{PCI}/device", "0x0") or "0x0")[2:]
    name = card_name(dev_id)
    prev = {}; t_prev = time.time()
    ring = collections.deque()                  # (t, card uJ, pkg uJ) over the last POWER_WINDOW s
    link = pcie_link(); link_t = time.time()
    while True:
        now = time.time(); dt = max(now - t_prev, 1e-3)
        cl = clients()
        busy = 0.0
        for cid, c in cl.items():
            p = prev.get(cid)
            if p and c["total"] > p["total"]:
                busy += (c["cycles"] - p["cycles"]) / (c["total"] - p["total"]) * 100
        e = int(rd(f"{h}/energy1_input", "0") or 0) if h else 0
        e2 = int(rd(f"{h}/energy2_input", "0") or 0) if h else 0
        ring.append((now, e, e2))
        while len(ring) > 2 and now - ring[1][0] >= POWER_WINDOW:
            ring.popleft()
        t0, ea, eb = ring[0]
        span = now - t0
        power = (e - ea) / 1e6 / span if span >= 2.5 and e >= ea else None
        pkg_power = (e2 - eb) / 1e6 / span if span >= 2.5 and e2 >= eb else None
        if now - link_t > 30:                   # the link can retrain (power saving, errors): re-read it
            link = pcie_link(); link_t = now
        temps = hw_temps(h) if h else {}
        vram_ch = [v for k, v in temps.items() if k.startswith("vram_ch_")]
        cap = rd(f"{h}/power1_cap") if h else None
        used_mb = sum(c["vram_kb"] for c in cl.values()) // 1024
        out = {
            "name": name, "pci": PCI, "ts": now,
            "vram_used_mb": used_mb, "vram_total_mb": TOTAL_MB,
            "busy_pct": round(min(busy, 100), 1),
            "temp_pkg": temps.get("pkg"), "temp_vram": temps.get("vram"),
            "power_w": None if power is None else round(power, 1),
            "pkg_power_w": None if pkg_power is None else round(pkg_power, 1),
            "power_cap_w": round(int(cap) / 1e6) if cap and cap.isdigit() and int(cap) > 0 else None,
            "temp_vram_max": max(vram_ch) if vram_ch else None,
            "temp_pcie": temps.get("pcie"), "temp_mctrl": temps.get("mctrl"),
            "pcie": link,
            "host_load1": round(os.getloadavg()[0], 2), "host_cpus": os.cpu_count(),
            "energy_j": round(e / 1e6, 1),           # cumulative card energy (hwmon energy1 = "card", uJ)
            "fan_rpm": int(rd(f"{h}/fan1_input", "0") or 0) if h else None,
            "freq_mhz": int(rd(glob.glob(f"/sys/bus/pci/devices/{PCI}/tile0/gt0/freq0/act_freq")[0], "0") or 0)
                        if glob.glob(f"/sys/bus/pci/devices/{PCI}/tile0/gt0/freq0/act_freq") else None,
        }
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f: json.dump(out, f)
        os.chmod(tmp, 0o644); os.replace(tmp, OUT)
        prev, t_prev = cl, now
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
