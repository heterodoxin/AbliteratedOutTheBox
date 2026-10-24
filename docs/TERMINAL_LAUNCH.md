# `strata launch`: set up, serve and watch, in the terminal

One command for the whole path: choose a model, tune it, start serving, and watch it. It is the same questions as the one-click setup (`START-HERE.bat` / `setup.sh`),
in the Terminal's look, with presets, an Advanced table, an access choice and a Connect view for your coding apps. Text previews of every screen are in
[docs/terminal-previews/](terminal-previews/) (`launch-*.txt`).

## Install (one line)

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh          # Linux, macOS
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                # Windows PowerShell
```

It finds out which computer this is, downloads the Terminal archive named in the release folder's `install-manifest.txt`, checks its SHA-256, unpacks it into your
home (`~/.local/share/strata/terminal`, `%LOCALAPPDATA%\Strata\terminal`; no administrator rights), checks the release's signed feed with the Terminal it just unpacked
(`strata update --check`; if that fails it removes what it unpacked), puts `strata` on your PATH and prints `Installed. Run: strata launch`. The engine (the part that talks
to your graphics card) comes with the first model: `strata launch` fetches the one for your card from the same release folder, signed, and test-starts it before using it.
The release folder is the latest GitHub release of Niko1221/Strata (`https://github.com/Niko1221/Strata/releases/latest/download`). The script only fetches and checks; read it first if you like (it is the same file as `install.sh` / `install.ps1` in the repository's `packaging` folder and in the release). Set `STRATA_FEED` to another folder to try a local one; `packaging/make_local_feed.py` builds a test-signed one. How far to trust a piped installer is in [SECURITY.md](../SECURITY.md).

## The screens (`strata launch`)

1. **Start**: saved presets (name, model, context, parallel requests, tuned or not, last used) or "Set up from the beginning" or "Use model files I already have".
2. **Model**: the catalog of the app and `setup.py` (one source), with download size, what it needs (graphics memory + RAM) and "fits / tight / does not fit this computer".
   A warning, never a block. The n-gram table stays on your SSD and is not counted. `o` = your own files (a folder or a .gguf; checked with the fingerprint table).
   *(Cards: only with two or more usable graphics cards. Each card gets a role: M Main, H Helper, O Off, R back to the recommendation, D details. `--gpu-roles 1=main,0=helper` does the same from the command line; see [MULTI_GPU.md](MULTI_GPU.md).)*
3. **Folder**: where model files are stored, free space shown. For a model that is already installed (also one the desktop app installed) it starts where the model is, says
   "Already downloaded" and lists only what is really missing with its size (the draft layer, about 5.2 GB; the picture reader, 0.9 GB). It never says "download the model" when only
   a helper file is missing.
4. **Context** (8K to 512K, with the extra memory each costs) and the memory format (`m`: 8-bit or 4-bit).
5. **Pictures**: only for models that can read them. They are **on by default** (Yes is preselected; `--vision off` or "No" turns them off). It says what it adds (about 0.9 GB:
   the picture-reader file, downloaded with the model through setup's Control mode, the same path the app's "Read pictures" uses). The reader runs on the CPU
   for a card under 24 GB (a few seconds per picture, nothing taken from the card) and on the graphics card from 24 GB (about 1-1.4 GB kept free), on NVIDIA and AMD cards; the
   rule lives in one place, `strata_control/vision_reader.py`, for setup, `strata launch` and the app's catalog. `--vision gpu` / `--vision cpu` (or the Pictures step) switch it per model. It needs a picture reader beside the engine in the engine package; with an
   engine that has none, pictures default to off and the screen says so in plain words. A model that cannot read pictures stays off with no question.
6. **Parallel requests**: "fit to your card" (12 GB: 1, 20: 2, 24: 3, 32+: 4) or 1 to 4; a warning if it will not fit.
7. **Tuning**: quick (about 20 minutes) / thorough (up to 2 hours) / skip (`strata tune auto`; the "answers may change very slightly" tick is off). The screens suggest quick the first
   time (it is done once and kept); the command line skips tuning unless you add `--tune`.
8. **Advanced** (optional): every engine setting with default, tuned value and your value; edit, reset, add a setting by name (checked against the engine's own list; a warning, not a refusal);
   the exact command (it includes the context, memory format and picture choice you made above; it has its own scroll area: Tab, then the arrows) and the matching `strata launch` flags.
9. **Access**: this computer only (127.0.0.1), your network (LAN address, key), Cloudflare quick tunnel (needs `cloudflared`), Tailscale (needs it installed and connected); port; API key
   (optional, recommended beyond this computer: Strata makes one for you, and one keypress (`t`) turns it off. Without a key the server still starts and warns that anyone who can reach the port can use the model and its tools; `--no-api-key` does the same without the screens, and on a Cloudflare tunnel the warning is stronger. This computer only never needs a key).
10. **Save as preset** and launch. Then downloads (with progress), engine start, and the monitor. In a small window the summary shrinks so the three choices stay on screen,
    and the key line says what Enter will do. No web page opens by itself; only `--open` opens one.

Keys: arrows choose, Enter goes on, Esc goes back, `?` explains the screen, `q` leaves. Every screen says in plain words what it means and what to do next.

## After the start: the monitor

The Strata Monitor with the new **Overview** (key 1: model state bar, speed, GPU load, VRAM, temperature, power, PCIe, CPU, disk, context fill, experts in VRAM, system RAM, recent requests; read from the
server's `/metrics`, the same data as the browser Monitor), the other five views as before, and **7 Connect** (the seventh view: the arrow keys reach it like the others, and every view's hint says 1-7): the address, the key (hidden until `k`), and the setup for each app
(`c` copy, `w` write it for you after a backup, `h` check it: one prompt and one tool call sent the way that app sends them). The list keeps two things apart: **Vendor tested** (the app
version its maker tested the setup with: history) and **This server** (Not checked / Passed / Failed for your server, after `h`). `q` asks: keep serving in the background, or stop
(stopping unloads the model and frees the graphics memory).

In the classic Windows console (the Windows 10 default window) the monitor first switches that window's font to Cascadia Mono when it is installed (the console font API: no admin, nothing saved, gone with the window) and draws the full look; otherwise it draws a solid-block look: █ bars with a thin ▄ track, ▄ / █ sparklines, square boxes and the 16 palette colours (`•` instead of `●` when the font is not Consolas). It always shows one line: "For the full look, use Windows Terminal (built into Windows 11)". Windows Terminal, Linux and macOS are never touched. `STRATA_GLYPHS=unicode|ascii|solid` forces a look. With two or more graphics cards the Overview tiles show the busiest card (load), all cards (memory), the hottest card and the total power, a "Cards" strip lists each card (up to 4, then "press 5" for the Hardware view), and the CPU tile says "2 CPUs · 64 cores" from the server's new `sockets` figure.

## Without the screens (flags)

```sh
strata launch --preset "IQ3_XXS 128K"                                 # a saved preset (opens the monitor)
strata launch --model iq3_xxs --context 128k --parallel 4 --vision off --tune quick \
              --access lan --port 8080 --save-preset "Office" --no-tui   # serve and print the address
strata launch --model-path /data/models/IQ3_XXS --no-tui               # files you already have
strata launch --preset Office --set prefill=65536 --set STRATA_PIN_BY_NEED=1 --no-tui
strata launch --model q2_0 --pool-cores 2-14:2 --no-tui               # experimental: pin the CPU workers to these logical processors (see DETAILS.md)
strata launch --plan --model q2_0                                      # what would happen; nothing changes
strata launch --model q2_0 --yes --no-tui                              # no questions (for one: download on a computer with no engine yet)
strata launch --model q2_0 --open                                      # also open the server's page in your browser once it serves
strata launch --model q2_0 --context 64k --restart --no-tui           # a model is already serving other choices (or holds the port): stop it and start these
strata launch --list-presets | --delete-preset NAME | --status | --stop [--preset NAME] | --show-key
```

`strata launch --help` lists everything with examples. An unknown model or flag gets a plain message with the closest match. Presets are small files in `launch-presets/` beside the profiles
folder; the run uses the shared profile `launch-<name>` and the shared tuning presets (`strata presets`), so the app sees them too. Flags next to `--preset` change that start only
(add `--save-preset` to keep them). `--plan` shows the model (also for your own files, which it recognises), the context, the picture setting and what would be downloaded and how big (0 when everything is already here). A model that was tuned before for the same choices on the same computer is not measured again by `--tune quick`; the stored result is used and the run says so. Add `--retune` to measure again (the model of the same launch is stopped for the measuring and starts again with the result).

**A launch while a model is already serving.** The same choices: nothing restarts, the running addresses are printed. Different choices (another context, pictures, access, or another model on the same port): the screens ask once, "Restart the model with these settings? Answers in progress stop.", with Restart and Cancel; `--no-tui` refuses with one line naming the running model, its port and settings, and says to add `--restart` or run `strata launch --stop` first. `--yes` alone never restarts. With `--restart` the old server is stopped first and the new one starts on the same port. A different port given with `--port` runs beside the first one; two big models may not fit in the graphics memory together, and the launch says so.

## Connect your apps: `strata connect`

`strata connect` lists OpenCode, Codex CLI, Claude Code, oh-my-pi (omp), pi, Hermes Agent, OpenClaw and the DeepSeek Harness (built into the app). `strata connect opencode` prints the file and
environment for the running server (address, key, model name and context filled in); `--write` writes it after asking and keeps a backup of anything it replaces; `--check` sends one prompt and one tool call
the way that app does and says whether each worked. With `--output json` (`strata connect opencode --check --output json`) it prints one JSON object with the same file and environment fields plus `checks` (one entry per check: `name`, `ok`, `detail`, `seconds`) and `checked_server`; `ok` is true only when both checks passed. The exit code is 0 when both worked and 1 when one failed or the server could not be reached; in that case the object also carries an `error` that says which of the two it was. Each configuration was tested for real against a Strata server at 32K context (the app versions are on screen); the known limits are printed in plain
words (for example: Claude Code needs `CLAUDE_CODE_MAX_CONTEXT_TOKENS`; OpenCode needs `limit.input`; Hermes refuses windows under 64K unless set up as the config does; OpenClaw needs Node 24.16+).

## Boxes the installer cannot set up

`setup.py` sets up NVIDIA (CUDA) and AMD (HIP) cards. For an Intel Arc card its Linux path is a container build that the Terminal does not run: if the model's files and their prepared pack are
already on the disk, `strata launch` writes the run settings itself and says so; otherwise it says which file is missing. Nothing is downloaded or compiled in that case.

## Models you already have

`strata models list` looks in the same places the screens and `strata launch` look (one function, `strata_control/discovery.py`): the folder you name with `--install-dir` or
STRATA_INSTALL_DIR (then only that one), the Terminal's own folder, the folders earlier installs remembered in settings.json, and the desktop app's install folder, which the Terminal
shares (Windows `%LOCALAPPDATA%\Strata\install`; Linux and macOS `$XDG_DATA_HOME/strata/install`, by default `~/.local/share/strata/install`). The data folder is the one you set, else the
app's (`app-settings.json` beside its install folder, else a `Documents/Strata` that holds models), else Strata's default. `strata models import PATH --plan` knows the Gyro-S and Gyro-M files
(TQK6/TQK7 and TQ2_T experts) and every file set setup installs, also under another name when the header matches, and says which run config, prepared pack, draft layer and picture reader
already exist for them. Nothing is copied or moved.

## When there is no engine for this computer

`strata launch` does not refuse. It says, in plain words, "Strata couldn't find an engine for this computer (the reason). The model will download, but it may not run until you install the right
engine (next step: ...)", asks whether to go on (Enter means yes; `--yes` skips the question; nothing to download means no question), downloads the model, and ends with a message that names the
engine problem and the next step. The exit code is 0 and nothing is started.

## Cards of different makes

One engine drives the cards of one make (CUDA: NVIDIA, HIP: AMD). Only those cards are offered for running and splitting; a model is never split across makes, and two models at the same
time, one on each make, is not supported yet. When the other make has the bigger card you get one line: "This computer also has an AMD R9700 (32 GB). The AMD build of Strata can use it - it fits bigger models."

## Limits

* The model download uses `setup.py` in its control mode (no pip, no engine fetch, no compiler), as the desktop app does; it needs internet and room for the files.
* `strata update`'s test start of an engine runs `<engine> --version`; the SYCL (Intel) package's engine does not answer that flag (exit 2), so on an Intel box the engine package has to be copied into the
  components folder until that is fixed.
* The Windows console has no way for the test scripts to drive the screens; they were checked by pilot tests and by hand.
