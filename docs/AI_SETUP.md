# Installing Strata for a user (instructions for AI agents)

This page is written for an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...) that a user asked to
set up Strata on their PC. Strata runs the Qwen3.8-Flash-Next model locally on one NVIDIA or AMD graphics card plus
system RAM (or on an Apple Silicon Mac), and serves an OpenAI- and Anthropic-compatible API on `http://127.0.0.1:8080`.
The human-oriented version of everything below is [INSTALL.md](INSTALL.md).

You use the **Terminal**: the `strata` command. Install it with one line, then `strata launch` downloads the engine
for the card and the model, and starts the server. If the Terminal cannot run on this PC, step 9 has the older route
from source.

Work through the steps in order. Tell the user what you are doing in plain words; they may not be technical.

## 0. Ground rules

- **Recommend an API key beyond this computer; do not force it.** Keep the default (`--access local`, the server listens on
  `127.0.0.1` only, no key needed). Only if the user asks for access from other devices, use `--access lan`. Strata makes a long
  random key for you then. Give the user the key (`strata launch --show-key`). If the user says they do not want a key, use
  `--no-api-key`: it starts and warns that anyone who can reach the port can use the model and its tools. Tell the user so.
  Do not put a tunnel or port forward in front of a server without a key unless the user asks for it.
- Do not change the user's system beyond what Strata does. The Terminal installs into the user's home folder and needs
  no administrator rights. It does not install drivers, Python packages or compilers. Ask before installing drivers.
- The model download is ~70 GB (94 GB for Unsloth's UD-IQ4_XS, 111 GB for UD-Q4_K_XL). Confirm the user is fine with
  that before you start, especially on a metered connection.
- The first `strata launch` is long-running (an hour or more on a slow connection). Run it in the background or with a
  long timeout and poll its output; do not kill it because it is quiet for a while. It is resumable: running the same
  command again continues where it stopped.
- Piping a script from the internet into a shell runs code you have not read. Tell the user what the line does (step 2).
  If they prefer, download the script first, show it, and run it only after they agree.

## 1. Check the PC

Requirements (details: [INSTALL.md](INSTALL.md#what-you-need)):

- **GPU:** NVIDIA RTX 20, 30, 40 or 50 series, or AMD Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT,
  RX 9070 / 9070 XT, Radeon AI PRO R9700, RX 6800 / 6900 series; 12 GB of VRAM or more (an NVIDIA card with 8 GB runs,
  slowly). GTX 10 series and older, and integrated GPUs, are not supported; Pascal / Volta cards (P40, V100) and some
  older AMD cards have experimental paths the user opts into ([OLDER_GPUS.md](OLDER_GPUS.md)). A **Mac** needs Apple
  Silicon (M1 or newer), macOS 15 or newer and 16 GB of memory or more ([MAC.md](MAC.md)).
- **Driver:** NVIDIA 580 or newer. AMD on Linux: the kernel's amdgpu driver; on Windows: a current AMD Adrenalin
  driver. If the driver is missing or too old, tell the user to update it (NVIDIA App / nvidia.com/drivers, or AMD
  Software) and restart; do not install drivers yourself unless they ask.
- **RAM:** 32 GB or more (see step 3). **Disk:** ~80 GB free, ideally on an NVMe SSD. **CPU:** x86-64 with AVX2.
  Without AVX2 (Xeon E5 v1/v2 and older) the ready-made engine refuses to start; the route from source builds an
  experimental and slow engine on the PC (10-20 minutes): tell the user that before starting
  ([INSTALL.md](INSTALL.md#older-cpus-experimental)).
- **OS:** Windows 10/11, Linux x86-64 (Ubuntu 22.04/24.04 and Debian 12/13 are fully automatic), or macOS on Apple Silicon.

To check by hand:

| What | Windows (PowerShell) | Linux | Mac |
| --- | --- | --- | --- |
| OS | `[Environment]::OSVersion` | `cat /etc/os-release` | `sw_vers` |
| NVIDIA GPU, VRAM, driver | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` | the same | - |
| AMD GPU | `Get-CimInstance Win32_VideoController \| Select-Object Name, AdapterRAM` (AdapterRAM is capped at 4 GB; trust the model name) | `lspci \| grep -i -E 'vga\|display'`; VRAM: `cat /sys/class/drm/card*/device/mem_info_vram_total` (bytes); `rocm-smi` only if ROCm is installed | - |
| RAM | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB` | `free -g` | `sysctl -n hw.memsize` (bytes) |
| Free disk | `Get-PSDrive -PSProvider FileSystem` | `df -h .` | `df -h ~` |

If the PC does not meet them, say which part is missing and stop.

## 2. Install the Terminal

One line. It needs no administrator rights. It checks what kind of PC this is, downloads the Terminal archive, checks
its SHA-256, unpacks it into the user's home folder, checks the release's signed update feed, and puts `strata` on the
PATH.

```
Linux, Mac:           curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
Windows (PowerShell): irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal window afterwards (the PATH changed). Check it: `strata --version`.

## 3. Pick the model

`strata launch --plan` shows what Strata would do and changes nothing. With no `--model` it picks the model for this
PC's RAM and prints the exact command line (`same as strata launch --model ...`). Use it as the default, or pick
yourself by the PC's RAM (ask the user whether they mainly want it for code - then the Coder is a good pick at any RAM
size):

| RAM | `--model` (and `--family`) | Notes |
| --- | --- | --- |
| 32 GB | `--family coder --model iq1_m` | for code; with a 24 GB card Q2_0 and IQ2_XS also run (Strata picks the low-RAM mode) |
| 48 GB | `--model iq2_xs` | or `q2_0` (fastest) |
| 64 GB | `--model iq2_xs` (recommended) | `iq3_xxs` / `iq3_s` are slower and a bit better |
| 96 GB+ | `--model iq3_s` | `--family unsloth --model ud-iq4_xs` (~4-bit, 94 GB, part of the experts read from the SSD under ~80 GB of RAM); `ud-q4_k_xl` is experimental: NVIDIA only, NVMe SSD, 7-8.5 tokens/s on 64 GB |

`--family swift` (Swift 1.5, a fine-tune that thinks shorter; sizes IQ2_XS, IQ3_XXS, IQ3_S) is the
alternative to `qwen`. A model name is not case sensitive. More: [MODELS.md](MODELS.md).

## 4. Set up and serve, without questions

```
strata launch --plan --model iq2_xs            # what would happen; nothing is changed
strata launch --model iq2_xs --yes --no-tui    # do it: download the engine and the model, start the server
```

With `--no-tui` and `--yes` it asks nothing, prints the addresses when the server is ready, and returns. The server
keeps running in the background. `strata launch` with no `--model`, `--preset` or other choice needs a terminal for
its screens, so an agent always names the model. Add `--output json` for output a script can read.

The flags (all of them, with examples: `strata launch --help`):

| Flag | Meaning |
| --- | --- |
| `--model NAME` | the size: `q2_0`, `iq2_xs`, `iq3_xxs`, `iq3_s`, `iq1_m`, `ud-iq4_xs`, `ud-q4_k_xl`, ... (the error message for a wrong name lists them) |
| `--family NAME` | the model version when several have that size: `qwen`, `swift`, `coder`, `unsloth` |
| `--model-path PATH` | files the user already has: a folder of shards or one `.gguf` file |
| `--data-dir DIR` | where new model files go. Models you already have stay where they are and stay usable; add `--move-data` to move them into DIR |
| `--context SIZE` | `32k`, `64k`, `128k`, `256k` or a number of tokens |
| `--vision off\|on\|gpu\|cpu` | read pictures. On by default when the model can read them: the reader runs on the CPU for a card under 24 GB and on the GPU from 24 GB, on NVIDIA and AMD cards. `off` turns it off |
| `--parallel N` | requests answered at the same time, 1 to 8 (default: fit to the card) |
| `--gpu LIST` | card numbers, e.g. `0` or `0,1` (several cards share the model) |
| `--tune quick\|thorough\|skip` | find the fastest settings first (default here: `skip`) |
| `--access local\|lan\|cloudflare\|tailscale` | who can connect; beyond `local` a key is made for you |
| `--port N` | the server port (default 8080) |
| `--api-key KEY`, `--api-key-env NAME`, `--new-key` | the key clients must send |
| `--no-api-key` | share without a key (a warning, not a refusal; recommended only on a network you trust) |
| `--save-preset NAME` | keep these choices; `strata launch --preset NAME` starts them again |
| `--plan` | show what would happen and exit |
| `--yes`, `--no-tui` | no questions, no screens |
| `--status`, `--stop`, `--show-key` | what is serving, stop it, print the key |

Notes:
- **No engine for this PC.** `strata launch` does not refuse. It says "Strata couldn't find an engine for this computer",
  goes on to download the model (with `--yes` it does not ask), and ends without starting anything (exit code 0; with `--output json` the answer
  has `"serving": false` and an `engine_problem`). Tell the user which card was not recognised and stop. Updating the
  engine later with `strata update` can fix it.
- **Exit codes:** 0 done, 2 a flag or name is wrong, 5 the Terminal or the engine cannot do it, 6 an install or the
  server start failed (the engine log is named in the message), 130 stopped.
- **Linux:** nothing needs `sudo`: the Terminal does not install packages. **Windows:** the first start may ask once
  whether to allow Strata on the network if you used `--access lan`.
- **AMD on Windows:** only the AMD driver is needed. The AMD engine is a ready-made download. It has not been run with a
  model on a real AMD card on Windows by us: ask the user to report how it runs ([AMD_HIP.md](AMD_HIP.md)).
- **Mac:** the Metal engine is fetched like any other. Pictures are on by default and read on the Mac's GPU ([MAC.md](MAC.md)).

## 5. While it downloads, tell the user

- It downloads about 70 GB from Hugging Face; how long depends on their connection (at 100 Mbit/s roughly 1.5-2
  hours). It can be stopped and continues where it left off.
- Then it prepares the model for their PC (a few minutes).
- **When the model starts, the PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
  loads 35-55 GB into RAM and locks part of it for the graphics card. That is normal; they should not close the window.
- Nothing leaves their PC: the model runs locally.

## 6. Check that it is serving

`strata launch` prints `Strata is serving.` with the addresses when it is done. Later:

```
strata launch --status          # lists what is serving, with its address
strata launch --stop            # stops it and frees the graphics memory
strata launch --preset NAME     # starts a saved preset again (after --save-preset NAME)
```

No browser opens by itself. Add `--open` to `strata launch` if the user wants the web page.

## 7. Verify

The HTTP server answers once the model is loaded (30-90 s on later starts, a few minutes the first time). Poll:

```
curl http://127.0.0.1:8080/health
```

It returns `{"status": "ok", "model": ..., "max_context": ..., "images": ..., "api_key": ..., "loaded": true, ...}`.
Then:

```
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"strata\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 64, \"reasoning_effort\": \"none\"}"
```

With an API key, add `-H "Authorization: Bearer <key>"`.

## 8. Connect the user's apps

- **Browser:** `http://127.0.0.1:8080` - Chat, a live Monitor, and About (settings and addresses).
- **Any OpenAI-compatible app or agent:** base URL `http://127.0.0.1:8080/v1`, any API key (or the configured one),
  any model name.
- **Anthropic-compatible apps:** base URL `http://127.0.0.1:8080`, without `/v1`. Those apps add `/v1/messages`
  themselves.
- **Codex CLI and other Responses API apps:** `http://127.0.0.1:8080/v1/responses` (stateless; Codex's
  `config.toml`: [DETAILS.md](DETAILS.md#the-responses-api-and-codex-cli)).
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (Strata ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the configured key).
- **The easy way for a coding app:** `strata connect` lists them (OpenCode, Codex CLI, Claude Code, oh-my-pi, pi,
  Hermes Agent, OpenClaw). `strata connect <app>` prints the file and environment for the running server,
  `strata connect <app> --write` writes it after asking (and keeps a backup), `strata connect <app> --check` sends
  one prompt and one tool call the way that app does and says whether each worked.
- **Thinking level:** `"reasoning_effort": "none" | "low" | "medium" | "high"` (default high).
- Strata answers one request at a time; `--parallel N` in `strata launch` decodes up to N together
  ([BATCHING.md](BATCHING.md)). API details: [DETAILS.md](DETAILS.md#using-it).

## 9. If the Terminal cannot run: install from source

Use this only when step 2 does not work on the PC (for example a CPU without AVX2, or no internet access to GitHub
Releases). It needs git or the zip, and on Linux `sudo` for Python and build tools.

```
git clone https://github.com/Niko1221/Strata.git
```

Without git: download https://github.com/Niko1221/Strata/archive/refs/heads/main.zip and unzip it. Pick a drive with
enough free space: the model files go to a `Strata-data` folder next to the Strata folder (or `--data-dir <path>`
on another drive). Setup's own check prints the GPU(s), driver, RAM, CPU, free disk and whether this PC can run
Strata, and installs nothing except Python and the `.venv` it needs to run:

```
Windows:  START-HERE.bat --check
Linux:    ./setup.sh --check
```

Run setup without questions, from the Strata folder (Windows: in `cmd`, or `cmd /c START-HERE.bat ...` from PowerShell).
The model is picked as in step 3, with setup's flag names (`--family qwen|swift|coder|unsloth --model Q2_0|IQ2_XS|...`):

```
Windows:  START-HERE.bat --yes --family qwen --model IQ2_XS --vision yes --no-start
Linux:    ./setup.sh --yes --family qwen --model IQ2_XS --vision yes --no-start
```

The flags (all of them: `START-HERE.bat --help`):

| Flag | Meaning |
| --- | --- |
| `--yes` | take the recommended answer to every question (no prompts) |
| `--family qwen\|swift\|coder\|unsloth` | the model version |
| `--model Q2_0\|IQ2_XS\|IQ3_XXS\|IQ3_S\|IQ1_M\|UD-IQ4_XS\|UD-Q4_K_XL` | the size (the Coder is IQ1_M, Unsloth UD-IQ4_XS or UD-Q4_K_XL) |
| `--context N` | context in tokens; default by VRAM: 32768 under 14 GB, 65536 under 20 GB, else 131072 |
| `--vision yes\|no\|gpu\|cpu` | read pictures. Pass `yes` to get the best reader for the card (the CPU under 24 GB, the GPU from 24 GB, any make); `no` turns them off |
| `--gpu N` / `--gpus 0,1` / `--gpus all` | one card, or several sharing the model (default: the card with the most VRAM) |
| `--backend cuda\|hip` | NVIDIA or AMD engine; chosen by itself on a PC with only one kind of card |
| `--data-dir PATH` | where new model files go. Nothing is moved unless you add `--move-data`; the saved default folder changes only with `--move-data` or `--set-default-data-dir` |
| `--port N` | the server port (default 8080) |
| `--host H --api-key K` | listen beyond this PC; only together with a key |
| `--no-start` | install only, do not start the server |
| `--setup` | install another model or change settings of an installed one |
| `--check` | only check the PC |

`--no-start` is recommended for agents: the server runs in the foreground until its window is closed, which would
block your shell. Start it separately:

```
Windows (PowerShell):  Start-Process -FilePath ".\run-iq2_xs.bat"            (opens its own window)
Linux:                 nohup ./run-iq2_xs.sh > strata-server.out 2>&1 &
```

Setup prints the start script it wrote (`start script: run-<model>.bat`). The name is the family tag plus the size,
lower case: `run-iq2_xs`, `run-swift-iq2_xs`, `run-coder-iq1_m`, `run-unsloth-ud-iq4_xs`. The engine log is
`strata-<model>.log` in the Strata folder. Then continue with step 7.

Notes:
- **Linux:** setup uses `sudo apt` (or dnf/pacman) to install Python with venv if it is missing, and on AMD may need
  `build-essential` and `git`. You cannot type the user's password: if a `sudo` step is needed, ask the user to run it
  (e.g. `sudo apt install python3-venv build-essential git`) and then rerun setup.
- **AMD on Linux:** setup installs ROCm into `.venv` (~10 GB, no sudo) unless a system ROCm 7 exists, and compiles the
  engine for the card (10-20 minutes, once).
- **AMD on Windows:** setup downloads the ready-made AMD engine (~550 MB, ROCm included) - nothing is compiled and only
  the AMD driver is needed. Before the model download it runs `engine\strata-device.exe --list-devices`; if that does
  not list the card, the driver is the problem (tell the user to update AMD Software). One card per model. Pictures use
  the 24 GB rule like everywhere else.
- **NVIDIA with no ready-made engine for the card:** setup offers to install build tools and compile (20-40 minutes);
  `--yes` accepts.
- **Mac:** `./setup.sh` builds the Metal engine on the Mac (needs Apple's Command Line Tools, `xcode-select --install`).

## 10. When something fails

| Symptom | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | The user updates the driver (NVIDIA App or nvidia.com/drivers) and restarts; run the command again. |
| Download or install stopped | Run the same command again; it continues. |
| `Strata couldn't find an engine for this computer` | The card is not one the engine packages cover, or the update feed was not reachable. Check the card against the list in step 1, check the internet, run `strata doctor`, then `strata update`. |
| `port 8080 is already in use` | Strata is already running (check `/health` and `strata launch --status`), or use `--port 8081`. |
| Very slow, disk busy, or "the engine stopped unexpectedly" | Not enough free RAM: close programs, or choose a smaller size (`q2_0` or `iq2_xs`). |
| `ExpertCache: cudaMalloc(...) failed: out of memory` with free VRAM (Windows) | The page file is off or tiny: set a fixed size (Custom size, the same initial and maximum size, e.g. 65536 MB) and restart; a page file Windows grows on demand ("System managed") may not grow in time (#60). |
| `prompt ... exceeds the context` | Start again with a bigger `--context`. |
| No AMD GPU found (Linux) | The amdgpu driver lists no GPU; check the card and driver. Integrated GPUs are not supported. |
| A Mac says the app is "damaged" (the app, not the Terminal) | The file is fine: [MAC.md](MAC.md) has the one command. |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and the [full table](DETAILS.md#troubleshooting). If it still fails,
run `strata doctor`, collect its output and the engine log named in the error message, and suggest an issue at
https://github.com/Niko1221/Strata/issues.

## Alternative: the MCP server

Strata also has an MCP server, so an AI tool can check, install, start and stop Strata through tool calls
(`strata_status`, `strata_models`, `strata_install`, `strata_start`, `strata_stop`, `strata_logs`) instead of the
shell commands above. See [MCP_SERVER.md](MCP_SERVER.md).
