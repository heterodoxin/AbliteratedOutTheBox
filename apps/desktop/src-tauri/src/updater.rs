//! The shell's half of the app update: Tauri's signed updater behind a loopback bridge for the App Host.
//!
//! The UI is served by the App Host, so it cannot call Tauri's updater plugin; and the host is Node, which has no updater.
//! The shell therefore opens one loopback listener, protected by a random token it hands only to the host it started
//! (`STRATA_SHELL_BRIDGE`, `STRATA_SHELL_TOKEN`), and the host's `app-updater.mjs` talks to it:
//!
//!   GET  /app-update/check    -> { available, version, current, notes }   (signature-verified metadata from the feed)
//!   POST /app-update/install  -> starts download + install in the background
//!   GET  /app-update/status   -> { state: idle|downloading|installing|done|error, downloaded, total, version, error }
//!
//! The update itself is Tauri's: it downloads the installer, checks its minisign signature against the public key in
//! `tauri.conf.json` (`plugins.updater.pubkey`, the owner's separate updater key) and installs. A check or an install happens
//! only when the user presses "Check for updates" / "Update Strata"; nothing here runs on start, and nothing here touches
//! engines or models (those are Control's component updater: `strata update`).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::AppHandle;
use tauri_plugin_updater::UpdaterExt;

/// Where an endpoint other than the one in `tauri.conf.json` may come from: a local test feed. Only loopback or https.
const ENDPOINT_ENV: &str = "STRATA_UPDATE_APP_ENDPOINT";

#[derive(Clone, Default)]
struct Progress {
    state: &'static str,
    downloaded: u64,
    total: u64,
    version: Option<String>,
    error: Option<String>,
}

impl Progress {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "state": if self.state.is_empty() { "idle" } else { self.state },
            "downloaded": self.downloaded, "total": self.total, "version": self.version, "error": self.error,
        })
    }
}

/// What the host needs to find the bridge.
pub struct Bridge {
    pub url: String,
    pub token: String,
}

fn endpoint_override() -> Option<String> {
    let url = std::env::var(ENDPOINT_ENV).ok()?;
    if url.starts_with("https://") || url.starts_with("http://127.0.0.1:") || url.starts_with("http://localhost:") {
        Some(url)
    } else {
        None
    }
}

fn updater_for(app: &AppHandle) -> Result<tauri_plugin_updater::Updater, String> {
    match endpoint_override() {
        Some(url) => {
            let url: tauri::Url = url.parse().map_err(|e| format!("bad update endpoint: {e}"))?;
            app.updater_builder().endpoints(vec![url]).map_err(|e| e.to_string())?.build().map_err(|e| e.to_string())
        }
        None => app.updater().map_err(|e| e.to_string()),
    }
}

async fn check(app: &AppHandle) -> Result<serde_json::Value, String> {
    let updater = updater_for(app)?;
    match updater.check().await.map_err(|e| e.to_string())? {
        Some(update) => Ok(serde_json::json!({
            "available": true, "version": update.version, "current": update.current_version, "notes": update.body,
        })),
        None => Ok(serde_json::json!({ "available": false, "current": app.package_info().version.to_string() })),
    }
}

async fn install(app: AppHandle, progress: Arc<Mutex<Progress>>) {
    let fail = |message: String| {
        let mut p = progress.lock().unwrap();
        p.state = "error";
        p.error = Some(message);
    };
    let updater = match updater_for(&app) {
        Ok(u) => u,
        Err(e) => return fail(e),
    };
    let update = match updater.check().await {
        Ok(Some(update)) => update,
        Ok(None) => return fail("There is no newer version of the Strata app.".into()),
        Err(e) => return fail(e.to_string()),
    };
    {
        let mut p = progress.lock().unwrap();
        *p = Progress { state: "downloading", version: Some(update.version.clone()), ..Default::default() };
    }
    let on_chunk = {
        let progress = progress.clone();
        move |chunk: usize, total: Option<u64>| {
            let mut p = progress.lock().unwrap();
            p.downloaded += chunk as u64;
            p.total = total.unwrap_or(p.total);
        }
    };
    let on_finish = {
        let progress = progress.clone();
        move || progress.lock().unwrap().state = "installing"
    };
    // Tauri checks the installer's minisign signature when it downloads it. Installing is done here, not by the plugin's
    // download_and_install: on Windows this app lives in a kill-on-close Job Object, and an installer the plugin starts is in
    // that job, so it is ended the moment the app exits and the update never happens. `launch_installer` starts it outside
    // the job; the app then exits and the installer (passive: a progress window, no questions) replaces it and starts it again.
    let bytes = match update.download(on_chunk, on_finish).await {
        Ok(bytes) => bytes,
        Err(e) => return fail(e.to_string()),
    };
    progress.lock().unwrap().state = "installing";
    match launch_installer(&update.version, &bytes) {
        Ok(()) => {
            progress.lock().unwrap().state = "done";
            app.exit(0);
        }
        Err(first) => match update.install(&bytes) {
            Ok(()) => {
                progress.lock().unwrap().state = "done";
                app.restart();
            }
            Err(e) => fail(format!("{first}; {e}")),
        },
    }
}

/// Windows: write the verified installer to a temp folder and start it detached and outside the app's Job Object.
#[cfg(windows)]
fn launch_installer(version: &str, bytes: &[u8]) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    let dir = std::env::temp_dir().join(format!("strata-app-update-{version}"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let installer = dir.join("installer.exe");
    std::fs::write(&installer, bytes).map_err(|e| e.to_string())?;
    std::process::Command::new(&installer)
        .args(["/P", "/R", "/UPDATE"])
        .creation_flags(CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("could not start the installer outside the app: {e}"))
}

/// Elsewhere the plugin's own install is used (the installer replaces the app in place).
#[cfg(not(windows))]
fn launch_installer(_version: &str, _bytes: &[u8]) -> Result<(), String> {
    Err("not Windows".into())
}

fn respond(stream: &mut TcpStream, status: &str, body: &serde_json::Value) {
    let text = body.to_string();
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
            text.len()
        )
        .as_bytes(),
    );
}

fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn handle(mut stream: TcpStream, app: &AppHandle, token: &str, progress: &Arc<Mutex<Progress>>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") && buf.len() < 8192 {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let head = String::from_utf8_lossy(&buf).to_string();
    let mut lines = head.lines();
    let request = lines.next().unwrap_or_default().to_string();
    let supplied = lines
        .find_map(|l| l.strip_prefix("authorization: ").or_else(|| l.strip_prefix("Authorization: ")))
        .map(|v| v.trim().trim_start_matches("Bearer ").to_string())
        .unwrap_or_default();
    if !same(&supplied, token) {
        return respond(&mut stream, "401 Unauthorized", &serde_json::json!({ "error": "not allowed" }));
    }
    let mut parts = request.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    match (method, path) {
        ("GET", "/app-update/check") => match tauri::async_runtime::block_on(check(app)) {
            Ok(value) => respond(&mut stream, "200 OK", &value),
            Err(e) => respond(&mut stream, "502 Bad Gateway", &serde_json::json!({ "error": e })),
        },
        ("POST", "/app-update/install") => {
            {
                let mut p = progress.lock().unwrap();
                if p.state == "downloading" || p.state == "installing" {
                    return respond(&mut stream, "409 Conflict", &serde_json::json!({ "error": "An app update is already running." }));
                }
                *p = Progress { state: "downloading", ..Default::default() };
            }
            tauri::async_runtime::spawn(install(app.clone(), progress.clone()));
            respond(&mut stream, "202 Accepted", &progress.lock().unwrap().json());
        }
        ("GET", "/app-update/status") => respond(&mut stream, "200 OK", &progress.lock().unwrap().json()),
        _ => respond(&mut stream, "404 Not Found", &serde_json::json!({ "error": "unknown route" })),
    }
}

/// Open the bridge on a free loopback port. Returns None if the listener cannot be bound (the app then simply has no
/// in-app update; the Terminal's `strata update --app` still works).
pub fn start(app: &AppHandle, token: String) -> Option<Bridge> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).ok()?;
    let port = listener.local_addr().ok()?.port();
    let progress = Arc::new(Mutex::new(Progress::default()));
    let app = app.clone();
    let secret = token.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            handle(stream, &app, &secret, &progress);
        }
    });
    Some(Bridge { url: format!("http://127.0.0.1:{port}"), token })
}

/// Environment for the App Host: where the bridge is, which app version this is.
#[allow(dead_code)]
pub fn apply_env(cmd: &mut std::process::Command, app: &AppHandle, bridge: Option<&Bridge>) {
    cmd.env("STRATA_APP_VERSION", app.package_info().version.to_string());
    if let Some(bridge) = bridge {
        cmd.env("STRATA_SHELL_BRIDGE", &bridge.url).env("STRATA_SHELL_TOKEN", &bridge.token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_comparison_needs_every_byte() {
        assert!(same("abc123", "abc123"));
        assert!(!same("abc123", "abc124"));
        assert!(!same("abc123", "abc12"));
        assert!(!same("", "x"));
    }

    #[test]
    fn progress_reports_idle_until_something_starts() {
        let idle = Progress::default().json();
        assert_eq!(idle["state"], "idle");
        let p = Progress { state: "downloading", downloaded: 10, total: 100, version: Some("0.2.0".into()), error: None };
        assert_eq!(p.json()["downloaded"], 10);
        assert_eq!(p.json()["version"], "0.2.0");
    }

    #[test]
    fn only_https_or_loopback_endpoints_may_override_the_configured_one() {
        for (value, ok) in [
            ("https://updates.example.com/latest.json", true),
            ("http://127.0.0.1:19080/app/latest.json", true),
            ("http://localhost:19080/latest.json", true),
            ("http://example.com/latest.json", false),
            ("file:///etc/passwd", false),
        ] {
            std::env::set_var(ENDPOINT_ENV, value);
            assert_eq!(endpoint_override().is_some(), ok, "{value}");
        }
        std::env::remove_var(ENDPOINT_ENV);
        assert!(endpoint_override().is_none());
    }
}
