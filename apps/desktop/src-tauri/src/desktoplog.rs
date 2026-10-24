//! The desktop shell's own logs (Linux): what the host and the window process printed, a crash log when the window dies, and
//! `--save-logs`, which collects everything a person can send us without the window (the window is what may be crashing).
//!
//! Files, in `<data>/logs` (`$XDG_DATA_HOME/strata/logs`, usually `~/.local/share/strata/logs`; the newest ten of each kind are kept):
//!   desktop-<UTC time>.log         one per start: the shell's lines, and every line the host (`[host]`) and the window (`[window]`) printed
//!   desktop-crash-<UTC time>.log   written when the window process dies without being closed: what happened in plain words, the system,
//!                                  the graphics session, the last lines of the window and of the host, and a backtrace when the system keeps one
//!
//! `strata-desktop --save-logs` puts the newest crash log, the tails of the newest shell logs and the engine server logs into one text file
//! in the Downloads folder. The same pieces go into Settings > Engine > Diagnostics > Save logs (the host reads this folder).
//!
//! Windows and macOS (one process owns the host and the window) use the same folder and file names for what happens while the app starts:
//! `init_startup` opens the run log first thing in `main`, before anything that can fail, and installs a panic hook; a start that fails
//! (the host does not come up, the window cannot be made, any panic) ends in `startup_failed`: the reason goes into the run log and a
//! `desktop-crash-` file, and the person sees a plain message box instead of a process that vanished with no trace. On Windows the folder
//! is `%LOCALAPPDATA%\Strata\logs`, next to the components folder.

// Linux uses the shell log, the crash log and --save-logs; Windows and macOS use the start-up half. Each side leaves the other unused.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const KEEP: usize = 10;
const RING_LINES: usize = 300;

/// `<data>/` : where Strata keeps its engine, runtime and logs on this system.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("STRATA_DESKTOP_DATA_DIR") {
        return PathBuf::from(dir);
    }
    platform_data_dir()
}

/// Windows: `%LOCALAPPDATA%\Strata`, the folder the host and Control already use for the components and the install.
#[cfg(windows)]
fn platform_data_dir() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join("AppData").join("Local")))
        .unwrap_or_else(std::env::temp_dir)
        .join("Strata")
}

/// Linux and macOS: `$XDG_DATA_HOME/strata`, usually `~/.local/share/strata`.
#[cfg(not(windows))]
fn platform_data_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("strata")
}

pub fn logs_dir() -> PathBuf {
    data_dir().join("logs")
}

/// "20261018-014503" in UTC, from the system clock (no time crate).
pub fn stamp(now: SystemTime) -> String {
    let secs = now.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil-from-days (Howard Hinnant), proleptic Gregorian
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Keep the newest `KEEP` files whose name starts with `prefix` and ends with `.log` (names sort by their time stamp).
fn prune(dir: &Path, prefix: &str, skip_crash: bool) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let mut names: Vec<String> = read
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix) && n.ends_with(".log") && !(skip_crash && n.starts_with("desktop-crash-")))
        .collect();
    names.sort();
    while names.len() > KEEP {
        let _ = std::fs::remove_file(dir.join(names.remove(0)));
    }
}

/// The last lines a process printed (kept in memory for the crash log).
#[derive(Clone, Default)]
pub struct Ring(Arc<Mutex<VecDeque<String>>>);

impl Ring {
    pub fn push(&self, line: String) {
        let mut q = self.0.lock().unwrap();
        if q.len() >= RING_LINES {
            q.pop_front();
        }
        q.push_back(line);
    }
    pub fn tail(&self, n: usize) -> Vec<String> {
        let q = self.0.lock().unwrap();
        q.iter().skip(q.len().saturating_sub(n)).cloned().collect()
    }
}

/// This run's log file, shared by the shell and the threads that copy the children's output.
#[derive(Clone)]
pub struct Log {
    file: Arc<Mutex<Option<std::fs::File>>>,
    pub path: Option<PathBuf>,
}

impl Log {
    pub fn open() -> Log {
        let dir = logs_dir();
        let _ = std::fs::create_dir_all(&dir);
        prune(&dir, "desktop-", true);
        prune(&dir, "desktop-crash-", false);
        let path = dir.join(format!("desktop-{}.log", stamp(SystemTime::now())));
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
        let log = Log { file: Arc::new(Mutex::new(file)), path: Some(path) };
        log.line("shell", &format!("Strata desktop {} starting (pid {})", env!("CARGO_PKG_VERSION"), std::process::id()));
        log
    }
    pub fn line(&self, who: &str, text: &str) {
        // never panics: this runs inside the panic hook too
        if let Ok(mut guard) = self.file.lock() {
            if let Some(f) = guard.as_mut() {
                let _ = writeln!(f, "[{who}] {text}");
            }
        }
    }
}

/// Print what the shell says (terminal and log).
pub fn say(log: &Log, text: &str) {
    eprintln!("[strata] {text}");
    log.line("shell", text);
}

/// Copy a child's output line by line: to our own stdout/stderr (as before), to the log file, and into the ring.
pub fn tee<R: Read + Send + 'static>(reader: R, who: &'static str, to_stderr: bool, ring: Ring, log: Log) {
    std::thread::spawn(move || {
        let mut r = BufReader::new(reader);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let text = String::from_utf8_lossy(&buf);
                    let text = text.trim_end_matches(['\r', '\n']);
                    if to_stderr {
                        eprintln!("{text}");
                    } else {
                        println!("{text}");
                    }
                    ring.push(text.to_string());
                    log.line(who, text);
                }
            }
        }
    });
}

fn first_line(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().and_then(|t| t.lines().next().map(str::to_string))
}

fn os_pretty_name() -> String {
    std::fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|t| t.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string())))
        .unwrap_or_else(|| "unknown".into())
}

/// The environment variables that decide how the window draws; nothing private.
fn graphics_environment() -> Vec<String> {
    const NAMES: &[&str] = &[
        "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP", "DESKTOP_SESSION", "DISPLAY", "WAYLAND_DISPLAY", "GDK_BACKEND", "GTK_THEME",
        "WEBKIT_DISABLE_DMABUF_RENDERER", "WEBKIT_DISABLE_COMPOSITING_MODE", "WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS", "LIBGL_ALWAYS_SOFTWARE",
        "GALLIUM_DRIVER", "__GLX_VENDOR_LIBRARY_NAME", "__NV_PRIME_RENDER_OFFLOAD", "MESA_LOADER_DRIVER_OVERRIDE", "APPIMAGE", "APPDIR",
    ];
    NAMES.iter().filter_map(|n| std::env::var(n).ok().map(|v| format!("  {n}={v}"))).collect()
}

/// The WebKitGTK files inside the AppImage (the version is in the file name or its link target).
fn bundled_webkit() -> String {
    let Some(appdir) = std::env::var_os("APPDIR") else { return "not an AppImage (the system's WebKitGTK is used)".into() };
    let lib = PathBuf::from(appdir).join("usr").join("lib");
    let mut found = Vec::new();
    if let Ok(read) = std::fs::read_dir(&lib) {
        for e in read.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with("libwebkit2gtk-4.1.so") || n.starts_with("libjavascriptcoregtk-4.1.so") {
                let target = std::fs::read_link(e.path()).map(|t| format!(" -> {}", t.display())).unwrap_or_default();
                found.push(format!("{n}{target}"));
            }
        }
    }
    found.sort();
    if found.is_empty() { "no WebKitGTK file found in the AppImage".into() } else { found.join("; ") }
}

/// A backtrace of the dead window process when the system keeps one (systemd-coredump); None when it does not.
fn system_backtrace(pid: u32) -> Option<String> {
    let mut child = std::process::Command::new("coredumpctl")
        .args(["info", &pid.to_string(), "--no-pager"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let deadline = std::time::Instant::now() + Duration::from_secs(4);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let mut out = String::new();
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    let out = out.trim().to_string();
    if out.is_empty() || out.contains("No coredumps found") { None } else { Some(out) }
}

/// What the crash log says, as text (so it can be tested without a crash).
#[allow(clippy::too_many_arguments)]
pub fn crash_report(what: &str, pid: u32, window_tail: &[String], host_tail: &[String], backtrace: Option<&str>, app_version: &str, log_path: Option<&Path>) -> String {
    let mut t = String::new();
    t.push_str("Strata app window crash\n=======================\n\n");
    t.push_str(&format!("When:     {} UTC\n", stamp(SystemTime::now())));
    t.push_str(&format!("What:     the app window process (pid {pid}) stopped without being closed: {what}\n"));
    t.push_str("Effect:   Strata kept running and opened in the web browser instead (browser mode); nothing was lost.\n");
    t.push_str(&format!("App:      Strata desktop {app_version}\n"));
    t.push_str(&format!("System:   {} / {}\n", os_pretty_name(), first_line("/proc/version").unwrap_or_default()));
    if let Some(v) = first_line("/proc/driver/nvidia/version") {
        t.push_str(&format!("NVIDIA:   {v}\n"));
    }
    t.push_str(&format!("WebKit:   {}\n", bundled_webkit()));
    t.push_str("Graphics session:\n");
    for l in graphics_environment() {
        t.push_str(&l);
        t.push('\n');
    }
    if let Some(p) = log_path {
        t.push_str(&format!("\nFull log of this run: {}\n", p.display()));
    }
    t.push_str("\n== Backtrace ==\n");
    match backtrace {
        Some(b) => t.push_str(b),
        None => t.push_str("none kept by this system (no systemd-coredump). To get one: run  coredumpctl gdb  or start the AppImage under  gdb --args ./<file>.AppImage --appimage-extract-and-run --window-child  and type  bt  after the crash.\n"),
    }
    t.push_str("\n\n== Last lines of the window process ==\n");
    for l in window_tail {
        t.push_str(l);
        t.push('\n');
    }
    t.push_str("\n== Last lines of the host ==\n");
    for l in host_tail {
        t.push_str(l);
        t.push('\n');
    }
    t
}

/// Write the crash log; returns where it is.
pub fn write_crash_log(what: &str, pid: u32, window: &Ring, host: &Ring, log: &Log) -> Option<PathBuf> {
    let dir = logs_dir();
    std::fs::create_dir_all(&dir).ok()?;
    prune(&dir, "desktop-crash-", false);
    let path = dir.join(format!("desktop-crash-{}.log", stamp(SystemTime::now())));
    let text = crash_report(what, pid, &window.tail(120), &host.tail(80), system_backtrace(pid).as_deref(), env!("CARGO_PKG_VERSION"), log.path.as_deref());
    std::fs::write(&path, text).ok()?;
    Some(path)
}

fn newest(dir: &Path, prefix: &str, crash: bool) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut names: Vec<String> = read
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix) && n.ends_with(".log") && (n.starts_with("desktop-crash-") == crash))
        .collect();
    names.sort();
    names.reverse();
    names.into_iter().map(|n| dir.join(n)).collect()
}

fn tail_lines(path: &Path, n: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// `--save-logs`: one text file with what we need to look at a crash. Returns its path.
pub fn save_logs(out_dir: &Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let path = out_dir.join(format!("strata-logs-{}.txt", stamp(SystemTime::now())));
    let mut t = String::new();
    t.push_str(&format!("Strata desktop logs {} UTC\nApp {}\nSystem: {} / {}\n", stamp(SystemTime::now()), env!("CARGO_PKG_VERSION"), os_pretty_name(), first_line("/proc/version").unwrap_or_default()));
    if let Some(v) = first_line("/proc/driver/nvidia/version") {
        t.push_str(&format!("NVIDIA: {v}\n"));
    }
    t.push_str(&format!("WebKit: {}\nGraphics session:\n{}\n", bundled_webkit(), graphics_environment().join("\n")));
    let dir = logs_dir();
    let crashes = newest(&dir, "desktop-crash-", true);
    t.push_str(&format!("\n\n##### Window crash logs ({} found, newest first, at most 3 shown)\n", crashes.len()));
    for p in crashes.iter().take(3) {
        t.push_str(&format!("\n----- {}\n{}\n", p.display(), std::fs::read_to_string(p).unwrap_or_default()));
    }
    let runs = newest(&dir, "desktop-", false);
    t.push_str(&format!("\n\n##### Shell logs ({} found, last 300 lines of the newest 2)\n", runs.len()));
    for p in runs.iter().take(2) {
        t.push_str(&format!("\n----- {}\n{}\n", p.display(), tail_lines(p, 300)));
    }
    let runtime = data_dir().join("runtime");
    if let Ok(read) = std::fs::read_dir(&runtime) {
        let mut files: Vec<PathBuf> = read.flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".server.log")).collect();
        files.sort();
        t.push_str(&format!("\n\n##### Engine server logs ({} found, last 200 lines each)\n", files.len()));
        for p in files {
            t.push_str(&format!("\n----- {}\n{}\n", p.display(), tail_lines(&p, 200)));
        }
    }
    std::fs::write(&path, t)?;
    Ok(path)
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Start-up failures (Windows and macOS)
// ---------------------------------------------------------------------------------------------------------------------------------

/// The run log and whether a failure may show a message box. Set once, first thing in `main`.
struct Run {
    log: Log,
    dialog: bool,
}

static RUN: OnceLock<Run> = OnceLock::new();
/// One message box per run: a failure inside the panic hook and the abort that follows must not show two.
static REPORTED: AtomicBool = AtomicBool::new(false);

/// This run's log, once `init_startup` has opened it.
pub fn run_log() -> Option<Log> {
    RUN.get().map(|r| r.log.clone())
}

/// Open the run log and install the panic hook. Call it first in `main`: nothing that can fail runs before it. `dialog` is false for
/// the self-test (no person to read a message box); `STRATA_NO_DIALOG` does the same for scripted runs.
pub fn init_startup(dialog: bool) {
    let dialog = dialog && std::env::var_os("STRATA_NO_DIALOG").is_none();
    let log = Log::open();
    log.line("shell", &system_line());
    if RUN.set(Run { log, dialog }).is_ok() {
        install_panic_hook();
    }
}

/// "Windows 10.0.26300 (64-bit)" or the name of the system; one line for the log and the crash file.
pub fn system_line() -> String {
    format!("System: {} {}", std::env::consts::OS, system_version())
}

#[cfg(windows)]
fn system_version() -> String {
    #[repr(C)]
    struct OsVersionInfo {
        size: u32,
        major: u32,
        minor: u32,
        build: u32,
        platform: u32,
        csd: [u16; 128],
    }
    #[link(name = "ntdll")]
    extern "system" {
        fn RtlGetVersion(info: *mut OsVersionInfo) -> i32;
    }
    let mut info = OsVersionInfo { size: std::mem::size_of::<OsVersionInfo>() as u32, major: 0, minor: 0, build: 0, platform: 0, csd: [0; 128] };
    // RtlGetVersion reports the real version (GetVersionEx lies to apps without a manifest entry)
    if unsafe { RtlGetVersion(&mut info) } == 0 {
        format!("{}.{}.{}", info.major, info.minor, info.build)
    } else {
        "version unknown".into()
    }
}

#[cfg(not(windows))]
fn system_version() -> String {
    first_line("/etc/os-release").unwrap_or_default()
}

/// What to tell a person, in plain words, for a technical reason. Always ends with the one next step.
pub fn plain_reason(detail: &str) -> String {
    let lower = detail.to_ascii_lowercase();
    if lower.contains("webview2") || lower.contains("webview") {
        "Strata shows its window with Microsoft's WebView2 component, and it could not be started. \
         Update or repair \"Microsoft Edge WebView2 Runtime\" (Settings > Apps > Installed apps), or install it from \
         https://go.microsoft.com/fwlink/p/?LinkId=2124703, then start Strata again."
            .into()
    } else if lower.contains("app host") || lower.contains("node runtime") || lower.contains("could not start the app host") {
        "Strata's background program did not start. Security software may have blocked a file in the Strata folder. \
         Allow the Strata folder in it (or reinstall Strata) and start Strata again."
            .into()
    } else {
        "Strata stopped while it was starting. Starting it again may work; if it does not, please send us the log file named below.".into()
    }
}

/// The text of the crash file for a failed start, as text (so it can be tested without a failure).
pub fn startup_report(detail: &str, app_version: &str, log_path: Option<&Path>) -> String {
    let mut t = String::new();
    t.push_str("Strata app start failure\n========================\n\n");
    t.push_str(&format!("When:     {} UTC\n", stamp(SystemTime::now())));
    t.push_str(&format!("App:      Strata desktop {app_version}\n"));
    t.push_str(&format!("{}\n", system_line()));
    t.push_str(&format!("What:     {detail}\n"));
    t.push_str(&format!("Meaning:  {}\n", plain_reason(detail)));
    if let Some(p) = log_path {
        t.push_str(&format!("\nFull log of this run: {}\n", p.display()));
    }
    t
}

/// Write the crash file of a failed start into `dir`; returns where it is.
pub fn write_startup_failure(dir: &Path, detail: &str, log_path: Option<&Path>) -> Option<PathBuf> {
    std::fs::create_dir_all(dir).ok()?;
    prune(dir, "desktop-crash-", false);
    let path = dir.join(format!("desktop-crash-{}.log", stamp(SystemTime::now())));
    std::fs::write(&path, startup_report(detail, env!("CARGO_PKG_VERSION"), log_path)).ok()?;
    Some(path)
}

/// The message box (Windows). Elsewhere the text goes to the terminal only.
#[cfg(windows)]
fn show_error(text: &str) {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(window: *mut std::ffi::c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
    }
    const MB_ICONERROR: u32 = 0x10;
    const MB_SETFOREGROUND: u32 = 0x1_0000;
    const MB_TOPMOST: u32 = 0x4_0000;
    let wide = |s: &str| std::ffi::OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (text, caption) = (wide(text), wide("Strata"));
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), caption.as_ptr(), MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST);
    }
}

#[cfg(not(windows))]
fn show_error(text: &str) {
    eprintln!("[strata] {text}");
}

/// Record a failed start (run log and crash file) and tell the person. Runs inside the panic hook as well, so it cannot panic.
fn report_failure(detail: &str) {
    let Some(run) = RUN.get() else {
        eprintln!("[strata] {detail}");
        return;
    };
    run.log.line("shell", &format!("START FAILED: {detail}"));
    let dir = logs_dir();
    let saved = write_startup_failure(&dir, detail, run.log.path.as_deref());
    if let Some(p) = &saved {
        run.log.line("shell", &format!("The crash file is {}", p.display()));
    }
    if run.dialog && !REPORTED.swap(true, Ordering::SeqCst) {
        let short: String = detail.chars().take(400).collect();
        let place = saved.or_else(|| run.log.path.clone()).map(|p| p.display().to_string()).unwrap_or_else(|| dir.display().to_string());
        show_error(&format!("Strata could not start.\n\n{}\n\nWhat happened: {short}\n\nA log was saved here, please send it to us:\n{place}", plain_reason(detail)));
    }
}

/// A start that cannot go on: say why (log, crash file, message box) and end the process with a failure code. Never returns.
pub fn startup_failed(detail: &str) -> ! {
    report_failure(detail);
    std::process::exit(1)
}

/// Every panic in the app (a Tauri setup error is one) ends in an abort with panic = "abort", which Windows reports as 0xc0000409 with
/// nothing else. The hook runs before the abort: it writes what the panic said and where, and shows the message box.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        previous(info);
        let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "a panic without a message".to_string()
        };
        let place = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_else(|| "an unknown place".into());
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        report_failure(&format!("the app stopped unexpectedly: {message} (at {place}, thread {thread})"));
    }));
}

pub fn downloads_dir() -> PathBuf {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(|h| PathBuf::from(h).join("Downloads")).unwrap_or_else(std::env::temp_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_stamp_is_utc_civil_time() {
        assert_eq!(stamp(UNIX_EPOCH), "19700101-000000");
        assert_eq!(stamp(UNIX_EPOCH + Duration::from_secs(1_800_000_000)), "20270115-080000");
        assert_eq!(stamp(UNIX_EPOCH + Duration::from_secs(1_760_751_903)), "20251018-014503");
    }

    #[test]
    fn the_ring_keeps_the_last_lines() {
        let ring = Ring::default();
        for i in 0..(RING_LINES + 50) {
            ring.push(format!("line {i}"));
        }
        let tail = ring.tail(3);
        assert_eq!(tail, vec![format!("line {}", RING_LINES + 47), format!("line {}", RING_LINES + 48), format!("line {}", RING_LINES + 49)]);
        assert_eq!(ring.tail(10_000).len(), RING_LINES);
    }

    #[test]
    fn the_crash_report_says_what_happened_in_plain_words_and_has_the_tails() {
        let text = crash_report("signal 11 (segmentation fault)", 4242, &["window said a".into(), "window said b".into()], &["host said x".into()], None, "0.2.0-beta", Some(Path::new("/x/desktop-1.log")));
        for needle in ["signal 11 (segmentation fault)", "pid 4242", "opened in the web browser instead", "0.2.0-beta", "window said b", "host said x", "/x/desktop-1.log", "none kept by this system"] {
            assert!(text.contains(needle), "{needle}");
        }
        let with = crash_report("x", 1, &[], &[], Some("#0 AcceleratedBackingStore::update"), "1", None);
        assert!(with.contains("#0 AcceleratedBackingStore::update") && !with.contains("none kept"));
    }

    #[test]
    fn logs_are_pruned_and_saved_into_one_file() {
        let dir = std::env::temp_dir().join(format!("strata-desktoplog-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("STRATA_DESKTOP_DATA_DIR", &dir);
        let logs = logs_dir();
        std::fs::create_dir_all(&logs).unwrap();
        for i in 0..14 {
            std::fs::write(logs.join(format!("desktop-20260101-0000{i:02}.log")), format!("run {i}\n")).unwrap();
            std::fs::write(logs.join(format!("desktop-crash-20260101-0000{i:02}.log")), format!("crash {i}\n")).unwrap();
        }
        prune(&logs, "desktop-", true);
        prune(&logs, "desktop-crash-", false);
        assert_eq!(newest(&logs, "desktop-", false).len(), KEEP);
        assert_eq!(newest(&logs, "desktop-", true).len(), KEEP);
        assert!(newest(&logs, "desktop-", false)[0].to_string_lossy().ends_with("000013.log"), "the newest are kept");
        std::fs::create_dir_all(dir.join("runtime")).unwrap();
        std::fs::write(dir.join("runtime").join("app-1.server.log"), "engine line\n").unwrap();
        let saved = save_logs(&dir.join("out")).unwrap();
        let text = std::fs::read_to_string(saved).unwrap();
        assert!(text.contains("crash 13") && text.contains("run 13") && text.contains("engine line"));
        std::env::remove_var("STRATA_DESKTOP_DATA_DIR");
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn a_failed_start_is_explained_in_plain_words_and_written_to_a_crash_file() {
        let webview = plain_reason("Failed to setup app: error encountered during setup hook: WebView2 error: 0x8007139F");
        assert!(webview.contains("WebView2") && webview.contains("go.microsoft.com"));
        assert!(plain_reason("the Strata App Host did not become healthy at http://127.0.0.1:47831/api/health").contains("background program"));
        assert!(plain_reason("something else").contains("log file"));
        let dir = std::env::temp_dir().join(format!("strata-startlog-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = write_startup_failure(&dir, "the app stopped unexpectedly: boom (at main.rs:1, thread main)", Some(Path::new("/x/desktop-1.log"))).unwrap();
        assert!(path.file_name().unwrap().to_string_lossy().starts_with("desktop-crash-"));
        let text = std::fs::read_to_string(&path).unwrap();
        for needle in ["boom", "main.rs:1", "/x/desktop-1.log", "0.2.0", "System:", "Meaning:"] {
            assert!(text.contains(needle), "{needle}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn on_windows_the_logs_sit_in_the_local_app_data_strata_folder() {
        let dir = platform_data_dir();
        assert!(dir.ends_with("Strata"), "{}", dir.display());
        assert!(system_version().split('.').count() == 3, "{}", system_version());
    }
}
