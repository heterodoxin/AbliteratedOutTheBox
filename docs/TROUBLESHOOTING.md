# Something went wrong?

The common problems and what to do, for the desktop app, the Terminal (`strata launch`) and the installer. Back to the [README](../README.md#something-went-wrong). The full table of
error messages, with the older engine fixes, is in the [details](DETAILS.md#troubleshooting).

## The desktop app and the Terminal

**A warning says Strata "couldn't find an engine for this computer".**
The engine is the part that talks to your graphics card. Strata only warns and still downloads the
model, and the model files stay on your computer, but the model may not run until the right engine is installed. The usual
reasons are that no engine is installed yet and the update source has none for your card, that the engine found is for
another make of card, or that the graphics driver is too old. In the app, open Settings, then Engine. In the Terminal, the
message names the next step: run `strata update` and start again. If your computer has no NVIDIA or AMD graphics card
(only an Intel UHD or Iris chip, an AMD processor graphics chip, a virtual machine, or Windows on ARM), Strata says so and
there is nothing to install: it needs a computer that has one. The model files stay where they are.

**"This model needs a newer engine." (the app), or a button called Update engine.**
Some models need an engine newer than the one installed: Gyro needs 0.1.45.1 or newer, NVFP4 needs 0.1.44, and NVFP4 (NVIDIA,
trained for FP4) needs 0.1.45. In the app, open Settings, then Engine, and press Update engine. It asks first, keeps your
model files and settings, and wants running models unloaded. In the Terminal: `strata update --check` says what would
change, `strata update` installs it (it switches to the new engine only after a test start). If the engine you
installed is the newest and the message stays, the update source has no newer engine for your card yet.

**On a Mac: "Strata is damaged and can't be opened", "cannot be opened", or "unidentified developer".**
Nothing is wrong with the file. macOS marks everything a browser downloads, and only lets a marked program run if Apple
has checked it. Strata's release builds are checked by Apple ("notarised"): drag Strata onto Applications and open it; the
first time, macOS asks "Are you sure you want to open it?" and you press Open. A test build that is not notarised says
"damaged". Do not move it to the Trash. Open the Terminal app and remove the mark, with the line that matches what you have
(drag the file from Finder into the Terminal window to type its path):

    xattr -dr com.apple.quarantine ~/Downloads/<the .dmg file>        # the .dmg, before opening it
    xattr -dr com.apple.quarantine /Applications/Strata.app           # the app, after dragging it over
    xattr -dr com.apple.quarantine <the folder you unzipped>          # an unzipped Terminal folder

Then open it again. If macOS says only "the developer cannot be verified", you can instead try to open it once, then open
System Settings > Privacy & Security and press "Open Anyway" next to the message about Strata. Installs made with
`curl ... | sh` carry no mark and need none of this. To see which kind you have: `spctl -a -vv /Applications/Strata.app`
answers "accepted" and "Notarized Developer ID" for a notarised build. (How the builds are signed: `packaging/mac/README.md`.)

**Checking what is installed: `strata doctor`.**
It is a read-only check of the installed engine parts (it changes nothing): which engine and helper files a saved model
needs, and whether they are there. `strata doctor --live` also asks the computer which graphics cards it sees and compares
them with the cards the model is set to use; `--output json` prints the same as data. Put its output into a bug report.

**Sending logs for a bug report.**
In the app: Settings, Engine, Diagnostics, View logs, then Save logs. It writes one text file,
`strata-diagnostics-<date>.txt`, into your Downloads folder and shows where. It holds the engine's messages, the failed
steps and the end of each model's server log. In the Terminal, send the output of `strata doctor`. If the Linux app window
crashes or will not open, run the AppImage with `--save-logs` in a terminal instead; it writes the crash log and the other
logs to one file in Downloads. If the Windows app closes right after you start it, it shows a message with the reason and writes
`desktop-crash-<date>.log` in `%LOCALAPPDATA%\Strata\logs`; Save logs includes it, and so can you, by hand. Look through a log before you send it, and never send an API key.

## While installing or starting

**My PC froze, or got very slow, the first time Strata started.**
That's normal while it starts, most of all the first time. Strata loads 35-55 GB into your RAM, locks part of it for
the graphics card, and works out how much of the model fits on your GPU. The mouse can freeze for a few minutes.
**Wait, and don't close the window.** The next starts are much faster. Still frozen after 10 minutes? Restart the
PC, close other programs (browsers use a lot of RAM) and try again. If it keeps happening, pick a smaller size (Q2_0
or IQ2_XS).

**The engine stops while it starts, on a small graphics card or a PC with little RAM. Can I turn the draft layer (MTP) off?**
Yes. The draft layer is what makes answers faster, and it takes about 1 GB of graphics memory (plus some RAM and a memory
copy that grows with the context). Without it the expert cache gets that memory back and answers are slower. Turn it off
for one model: in the app, Models, Expert settings, "Draft layer (MTP)": Off; in the Terminal,
`strata launch --model q2_0 --mtp off` (older Terminals: `--set mtp=off --set spec-min-p=off`); by hand, add `"mtp": false` to the
model's `strata-<model>.json`. The choice is kept per model and applied on every start.

**I changed `strata-<model>.json` by hand and the change came back.**
Engine options you add to `args` by hand (control vectors, `--spec`, `--vram-reserve-mib 0`, anything the app does not offer)
are kept. The file has two helper keys for this. `user_args` lists your options; every start and every setup run puts them
over the ones the app generates, and for the same option your value wins. `args_written` is a copy of `args` as the app last
wrote it, so an option you added or changed in `args` is noticed and moved into `user_args` the next time the app or setup
touches the file. To undo one, delete it from both `args` and `user_args`. A value you save in the app's Expert settings
("GPU memory for other apps") is put in `user_args` for you.

Two things the app still chooses itself: the context size, KV type, picture reading and draft layer. Set them in the Models
screen, or with `"mtp": false` (draft layer). The app keeps the earlier file as `strata-<model>.json.bak`.

**The app keeps more graphics memory free than I want (decode is slower than in 0.1.45).**
Without a value of yours the app keeps 1536 MiB free (2048 with picture reading) on a card that drives a display, and 512 MiB
(700 with pictures) on a card that has no display attached, such as a second GPU with the screen on the integrated one. It
reads that from the driver (nvidia-smi `display_active`, or the Windows adapter that has the desktop); when it cannot tell, it
assumes a display. Your own value is never raised or lowered: set "GPU memory for other apps" in Expert settings, or
`--vram-reserve-mib 0` in `args`. The engine's own warning about low free memory stays; a card that also draws the desktop
can freeze or slow to a few tokens a second with a very small reserve.

**It stopped while downloading or installing.**
Run `START-HERE.bat` (Linux: `./setup.sh`) again. It continues where it stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run
`START-HERE.bat` again.

**It says port 8080 is already in use.**
Strata is already running. Look for its window. Or another program uses the port: `START-HERE.bat --port 8081`.
Only an address that is really taken says "already in use". Any other reason (Windows keeps the port reserved, or
the config's `"host"` is not an address of this PC) is printed as what the OS said, with what to change (#769).

**"Windows blocked the Strata engine (or the image encoder) ... Smart App Control".**
On a clean Windows 11, Smart App Control refuses programs that are not signed, and Strata's `strata.exe` and
`strata-vision.exe` are not (#735). Turn it off (Windows Security > App & browser control > Smart App Control
settings; it cannot be turned back on without a reset of Windows), or, if only the image encoder is blocked, run
`START-HERE.bat --setup --vision no`.

**A new engine misbehaves after an update.**
An update keeps the engine it replaced in `engine\.previous` (one generation, about 210 MiB).
`python setup.py --rollback-engine` puts it back (and keeps the newer one there: run it again to go forward) (#670).

**UPDATE.bat / update.sh say "git pull did not succeed", or "has no commit in common" (#1276).**
The repository's history was cleaned up on 2026-10-06, so a clone made before that cannot be updated with `git pull`.
The 0.1.40.2 scripts handle it: when no tracked file is edited they keep your old commits in the branch
`pre-cleanup-backup` and move the clone to the new history; otherwise they print the two commands for you. Your models,
settings and engine are untracked files and are never touched. An older UPDATE.bat / update.sh (before 0.1.40.2) cannot
do this itself: run these once in the Strata folder, then use UPDATE.bat as usual (add `git stash` first if `git status`
lists edited files):

    git fetch origin
    git branch pre-cleanup-backup
    git checkout -B main origin/main

**Python or the build tools could not be installed.**
Install what it names (links are printed), then run it again. Everything already done is kept.

**Linux: the engine does not compile (`unsupported GNU version`, or `exception specification is incompatible` for
`cospi`/`sinpi`/`rsqrt`).** Two known mismatches between the CUDA toolkit and a new Linux (#601):
- gcc newer than 14 (Ubuntu 26.04's default 15): CUDA 12.x and 13.0 refuse it. Install g++-14 and run
  `CXX=g++-14 ./setup.sh` (CC, CXX and CUDAHOSTCXX name the build's compilers; setup passes them to CMake itself,
  so an already-configured build folder picks them up too, #1645).
- glibc 2.43 with CUDA 12.9: the toolkit's math headers clash with glibc's, already in CMake's first test compile.
  Use CUDA 12.8 or 13.x instead. Setup takes the newest toolkit it finds; `STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc
  ./setup.sh` makes it use that one (only that one).

**The first start takes minutes.**
It is reading 34-55 GB into RAM; the second start is faster while the files are in the OS cache. Started from Task
Scheduler, it can be 24x slower: see [Running it at startup](DETAILS.md#running-it-at-startup-task-scheduler).
On Linux the engine now asks the kernel to read the model files ahead (a cold start went from ~920 s to 70 s on one
PC); `STRATA_READ_AHEAD=0` turns that off. On Linux with transparent huge pages on `always`, the arena no longer asks
for `MADV_HUGEPAGE` on top (a fragmented machine spent minutes compacting memory, #771); `STRATA_NO_ARENA_THP=1` skips
that request on any setting.

**Setup says the engine does not match GitHub's checksum.**
Setup checks the downloaded ready-made engine against the SHA-256 GitHub publishes. On a mismatch the file is corrupt or was changed on the way, so setup deletes it and stops; run setup again to download it afresh. If GitHub gives no checksum (offline, rate limited, your own mirror), setup warns and installs anyway. `STRATA_SKIP_SHA256=1` skips the check, only if you insist.

## While it answers

**It's very slow and the disk light keeps blinking.**
Your PC is out of free RAM. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine). Just send your message again: Strata starts the
engine by itself. If it keeps happening, close other programs or pick a smaller size.

**On Linux the engine stops with "exit code -9", but nothing ran out of RAM.**
Check `ulimit -R`. If it prints 0 (or any number), the desktop session sets a real-time CPU limit: the audio system (PipeWire) does that when rtkit is not installed, and every program started from the desktop inherits it. The kernel can then end a program with signal 9 and no message. Install `rtkit`, or start Strata outside the session's limits, for example `systemd-run --user --collect -p LimitRTTIME=infinity -- strata serve ...`. Strata prints a warning at start when it sees this limit.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, or run `SETUP.bat` and pick more
context.

**It is slower than the tables.**
The monitor plugged into the graphics card and other GPU programs take VRAM from the expert cache; RAM running below
its rated speed (enable EXPO/XMP in the BIOS) slows the CPU half.

**An earlier reply that came back empty is gone from the chat history (0.1.40, #843).**
The server leaves out finished assistant turns that have no text, so the next reply does not copy the empty ones.
`STRATA_KEEP_EMPTY_TURNS=1` in the server's environment renders them as before.

**A Turing card (RTX 20) reads prompts above ~90K tokens differently (0.1.40, #743).**
The prompt's top-k selection takes a wider kernel there, with the same ids. `STRATA_TOPK_STREAM=0` restores the old one.

**Decode is several times slower with `--expert-cache N` than with `auto` (#781, #831).**
An explicit N is a byte budget that is not checked against the free VRAM once the slots are written. On a card it
fills (the log says `0 MiB of VRAM free with everything loaded - LOW`), Windows pages the GPU's memory and decode drops
from 100 to 14 tok/s with only a few hundred slots too many. Use `--expert-cache auto`, or a smaller N.

**A low-RAM PC (about 32 GB) stalls with `--resident-budget-gib N` (#649).**
The budget is pinned (locked) in RAM, which leaves the OS and the CPU workers little room. `STRATA_RESIDENT_PIN=0` keeps
it pageable; a smaller budget works too.

**Decode got slower in 0.1.40 on a 32 GB PC with `--resident-budget-gib N`, and the drive is read hard (#1194, #1116, #1085).**
The start line "the file tier reads unbuffered" means the experts outside the RAM copy are read straight from the drive.
0.1.40 chose that whenever the free RAM could not hold all of them; but the same few come back token after token, and
a cache a twentieth of their size serves most of the repeats. 0.1.40.2 asks for that much (at least 1.5 GiB beside the
4 GiB headroom) and reads through the cache otherwise, which measured 7% to 118% faster decode with a third to a sixth of
the drive traffic on a 20 to 32 GB box. On 0.1.40 / 0.1.40.1: `STRATA_UNBUFFERED_LOAD=0` does the same, `=1` forces the
unbuffered reads.

**Linux: the start is OOM-killed while the engine allocates the page-locked expert copy (#1250).**
Page-locked pages come from the driver in one go and cannot wait for the kernel to give back file cache, so with the
model files' cache in the way (little really free RAM, much "available") the host could run out. 0.1.40.2 gives back the
cached pages of the model files it mapped before it page-locks the copy; when the really free RAM still does not cover it,
it takes the pages in 1 GiB steps while the RAM available (cgroup limits included) stays above the headroom, page-locks
what it has, and keeps the rest resident but pageable with a warning ("only N of M GiB could be page-locked"): the answers
are the same, only the copies of the pageable part to the GPU go through the CPU. `STRATA_PIN_GUARD=0` turns this off,
`STRATA_PIN_RESERVE_GIB=N` sets the RAM that must stay free (default: `STRATA_RESIDENT_HEADROOM_GIB`, 4 GiB).

**The Windows display driver resets, then the PC blue-screens (0x141, then 0x116 in `nvlddmkm`) during an answer (#961).**
This is the NVIDIA driver, not Strata: the report shows the same wedge with other CUDA programs on that driver (610.88,
RTX 4080 SUPER). Try another driver (a Studio one, or an older one) and a lower power limit; if it only happens with
`--spec 4`, `--spec 0` avoids the verify window while you wait for a driver fix.

**An RTX 50 card (a source build with CUDA 13.2) answers with nonsense, or reads prompts wrongly (#892, #968).**
CUDA 13.2.0 and 13.2.1's compiler (nvcc 13.2.51) miscompile some of the engine's kernels for sm_120: on our RTX 5070 the IQ2_S and IQ3_S
products are wrong (relative error 0.5 to 1.0 in the tests) with 13.2 and right with 13.0. The ready-made engine is built
with 13.0. If you compile it yourself, use CUDA 13.0, 13.1 or 13.2.2 (13.2.2, nvcc build 13.2.86, fixes it; an older one can sit next to a newer: `STRATA_NVCC=<path to its nvcc>`);
setup warns when it finds 13.2.0 or 13.2.1 for such a card, and takes an older 13.x when one is installed.

**Pictures are refused, or slow.**
"this server was started without the vision encoder": the model was set up for text only (someone chose "No", or the
engine has no picture reader) - run setup again with `--vision yes` (or `--vision gpu` / `--vision cpu`), or switch pictures
on in the app or on the Pictures step of `strata launch`. Pictures are on by default wherever a model can read them. The
reader runs on the CPU under 24 GB of graphics memory and on the card from 24 GB, on NVIDIA and AMD cards. Pictures that take
several seconds (about 3 s at 300 image tokens on 8 cores, more with more tokens) are read by the encoder on the CPU;
`--vision gpu` (NVIDIA and AMD, ~1.4 GB of VRAM) makes it 0.1-0.5 s. On AMD the card's reader is `strata-vision-hip`; if it
cannot start, the server prints "the GPU picture reader did not start (...); pictures are read on the CPU" and goes on with
the CPU reader. On a Mac the reader runs on the GPU; `--vision cpu` takes about 50 s per picture there. On Intel Arc, pictures are off by default for now (`--vision yes` turns them on).

## AMD cards

**"No AMD GPU found (the amdgpu driver's KFD topology is empty)" on Linux.**
The kernel's amdgpu driver is not loaded for the card. Integrated Radeon GPUs are listed as not supported; setup
lists every card it found and whether Strata can use it.

**The engine stops at start with the card's name, its architecture and the build's list.**
The engine was compiled for another card (for example after moving the Strata folder to another PC). Run
`./setup.sh --setup --backend hip`: it compiles the engine for this card's architecture.

**On Windows the engine exits with `0xC0000005` in `amdhip64_7.dll` before it prints anything (#654).**
Two things in your environment can cause it, and the server now repairs both before it starts the engine (it says so in
the log): `HIP_PATH`, `HIP_DEVICE_LIB_PATH` or `LLVM_PATH` naming a ROCm folder that no longer exists (a deleted build),
and a `TEMP`/`TMP` folder the engine cannot create files in (AMD's runtime compiles its first kernels through temporary
files). If you start `strata.exe` by hand, fix them yourself: `set HIP_PATH=`, and point `TEMP` and `TMP` at a normal
folder such as `C:\Temp`.

**Windows AMD: the driver resets (VIDEO_ENGINE_TIMEOUT_DETECTED / screen flicker / the engine dies mid-answer).**
Windows gives a GPU about 2 seconds to answer; when a graphics card does not, it resets the driver (a TDR): the screen
flickers, the engine dies, and the Event Viewer shows `VIDEO_ENGINE_TIMEOUT_DETECTED` (#613, with #579 and #541 on
Linux). Strata 0.1.39 and newer say "the GPU stopped responding" in the log. It was seen with KV streaming on
(`--kv-resident 32768`, which setup turns on from 64K context) while reading a long prompt on a gfx1201 card. Try these
in order, one at a time, and tell us what changed:
1. **Send us the evidence.** The last ~80 lines of `strata-<model>.log` in the Strata folder (the server window shows
   the same lines), your AMD driver version (AMD Software > System), your card and Windows version, the Strata
   version, and the exact steps (context size, how long the prompt was, which app sent it). If Windows wrote a
   dump, the newest `.dmp` in `C:\Windows\LiveKernelReports\WATCHDOG` (or `C:\Windows\Minidump`) helps; it holds no
   chats, only the driver state.
2. **Find the step that hangs:** add `"STRATA_PF_STEP_SYNC": "1"` to the `"env"` block of `strata-<model>.json`
   and restart. The prompt path then waits for the GPU after each step and logs any step over 250 ms, so the log names
   the step. It is slower; use it to diagnose, then take it out.
3. **`"STRATA_KV_HOST_DMA": "1"`** in the same `"env"` block: no GPU kernel writes the RAM copy of the K/V during a
   prompt, it is copied by DMA instead. Same answers.
4. **No KV streaming:** run setup again and pick 32K context (or remove `--kv-resident` and its number from
   `"args"` in the JSON, with a context that fits the VRAM).
5. **A smaller prompt chunk:** change `--prefill` in `"args"` (for example `--prefill 512`): each GPU launch then does
   less work between two checks by Windows. Prompts are read more slowly.
6. **Optional, your decision: give Windows more time (`TdrDelay`).** The default is 2 seconds. In the registry,
   `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`, a DWORD `TdrDelay` of `10` (seconds)
   lets a long GPU job finish before Windows resets the driver; restart the PC. It hides the symptom and does not fix
   a GPU that has really hung, and a hung GPU then freezes the screen for those 10 seconds. Strata changes no Windows
   setting itself; to undo it, delete the value.
7. **Driver:** use the newest AMD Software release for your card (Adrenalin, not a Windows Update driver) and restart
   after installing. If it started after a driver update, tell us both versions; going back one version is a fair
   test.

**Large pinned host allocations fail on ROCm although RAM is free.**
See [AMD_HIP.md](AMD_HIP.md#model-and-serving-configuration): the mapped expert mode avoids the full pinned arena.

**"verify: timed out at layer N" or "no progress for 60 s" on AMD (Linux), mostly with `--mmap-experts`.**
Two reports fixed it with `"GPU_PINNED_MIN_XFER_SIZE": "1048576"` (RX 6800) or `"HSA_USERPTR_FOR_PAGED_MEM": "0"` (two
R9700) in the `env` block of the server JSON, then a restart. Experimental, one machine each; on a systemd unit use
`MemoryMax`, not `MemoryHigh`. Details: [AMD_HIP.md](AMD_HIP.md#linux-verify-timeouts-while-the-kernel-reclaims-host-memory-experimental-workarounds).

## Still stuck?

Look in the [full troubleshooting table](DETAILS.md#troubleshooting), or open an
[issue](https://github.com/Niko1221/Strata/issues) and attach the log: in the app, the file Save logs writes (see
[above](#the-desktop-app-and-the-terminal)); in the Terminal, the output of `strata doctor`; with the installer,
`strata-<model>.log` from the Strata folder. Say which one you use (app, Terminal or the installer), your Strata and
engine versions and your card.
