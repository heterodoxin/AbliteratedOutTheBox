// Linux file drop (found on z440 under Xvfb with a real GTK drag, WebKitGTK 2.52): the page gets no files and WebKit never raises the drop signal,
// so the shell reads the file:// addresses from the drag data, accepts the drag itself, takes the "leave" with every button up as the drop and
// hands the paths to the host. The behaviour itself is checked on a real X display (scripts in phaseA-tests); these keep the pieces in place.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = path => readFile(new URL(path, import.meta.url), 'utf8').then(s => s.replace(/\r\n/g, '\n'));

test('the Linux shell hooks the drag signals of the page and keeps Windows and macOS on their own paths', async () => {
  const rust = await read('../../desktop/src-tauri/src/main.rs');
  assert.match(rust, /#\[cfg\(target_os = "linux"\)\]\s+hook_linux_drops\(&window, port, shell_token\);/);
  assert.match(rust, /view\.connect_drag_data_received\(/); assert.match(rust, /file_uri_to_path\(u\.as_str\(\)\)/);
  assert.match(rust, /ctx\.drag_status\(gtk::gdk::DragAction::COPY, time\)/);
  assert.match(rust, /pointer_buttons_up\(widget\)/);
  assert.match(rust, /hand_over_drop\(target\.clone\(\), port, token\.clone\(\), list\)/);
  assert.match(rust, /fn file_uri_to_path\(uri: &str\) -> Option<String>/);
  assert.match(rust, /#\[cfg\(windows\)\]\s+let builder = builder\.disable_drag_drop_handler\(\);/);
  const toml = await read('../../desktop/src-tauri/Cargo.toml');
  assert.match(toml, /\[target\.'cfg\(target_os = "linux"\)'\.dependencies\]\ngtk = "0\.18"\nwebkit2gtk = "2\.0"/);
});

test('only file:// addresses reach the host, which applies the same limits as for every drop', async () => {
  const rust = await read('../../desktop/src-tauri/src/main.rs');
  assert.match(rust, /strip_prefix\("file:\/\/"\)\?/);
  const host = await read('../server/native-drop.mjs');
  assert.match(host, /MAX_FILES = 10, MAX_BYTES = 25 \* 1024 \* 1024/);
  assert.match(host, /!isAbsolute\(path\)/);
});
