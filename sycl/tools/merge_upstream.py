# merge_sel.py BASE_OUT NEW_OUT OLD_REV NEW_REV: 3-way merge into sycl/ of the files upstream changed OLD_REV..NEW_REV
import re, subprocess, sys, os, shutil
B, N, old, new = sys.argv[1:5]
pat = re.compile(r"\b([A-Za-z0-9_]+)_([0-9a-f]{6})\b")
H = lambda t: pat.sub(lambda m: m.group(1) + "_HASH", t)
def hashes(t):
    m = {}
    for a, h in pat.findall(t): m.setdefault(a, set()).add(h)
    return {a: next(iter(v)) for a, v in m.items() if len(v) == 1}
changed = subprocess.check_output(["git", "diff", "--name-only", old, new, "--", "src", "include"], text=True).split()
HAND = {"src/core/mtp.cpp", "src/core/verify.cpp", "src/ngram/ple_reader.cpp", "src/kernels/native_expert_parity.cpp"}
for f in changed:
    rel = f[:-3] + ".dp.cpp" if f.endswith(".cu") else f
    pf = "sycl/" + rel
    if f in HAND:
        if not os.path.exists(pf): print("HAND-MISSING", f); continue
        open("/tmp/_b", "w").write(subprocess.check_output(["git", "show", f"{old}:{f}"], text=True))
        open("/tmp/_n", "w").write(subprocess.check_output(["git", "show", f"{new}:{f}"], text=True))
        r = subprocess.run(["git", "merge-file", "-L", "port", "-L", "base", "-L", "new", pf, "/tmp/_b", "/tmp/_n"]).returncode
        print("HAND", "MERGED" if r == 0 else f"CONFLICT {r}", rel); continue
    bf, nf = f"{B}/{rel}", f"{N}/{rel}"
    if not os.path.exists(nf):
        print("NOT-MIGRATED (original used)" if not os.path.exists(pf) else "PORT-HAS-NO-DPCT?", rel); continue
    nt = open(nf).read()
    if not os.path.exists(pf):
        os.makedirs(os.path.dirname(pf), exist_ok=True); os.makedirs(os.path.dirname(pf), exist_ok=True); shutil.copy(nf, pf); print("NEWCOPY", rel); continue
    if not os.path.exists(bf):
        print("NO-BASE (new to dpct output)", rel); continue
    bt = open(bf).read()
    if H(bt) == H(nt): print("NOCHANGE-after-dpct", rel); continue
    pt = open(pf).read(); ph = hashes(pt); bh = hashes(bt)
    nt = pat.sub(lambda m: f"{m.group(1)}_{bh[m.group(1)]}" if m.group(1) in bh else m.group(0), nt)
    canon = lambda t: pat.sub(lambda m: f"{m.group(1)}_{ph[m.group(1)]}" if m.group(1) in ph else m.group(0), t)
    bt, nt = canon(bt), canon(nt)
    open("/tmp/_b", "w").write(bt); open("/tmp/_n", "w").write(nt)
    r = subprocess.run(["git", "merge-file", "-L", "port", "-L", "base", "-L", "new", pf, "/tmp/_b", "/tmp/_n"]).returncode
    print("MERGED" if r == 0 else f"CONFLICT {r}", rel)
