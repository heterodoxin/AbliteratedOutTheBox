# Signing and notarising the Mac builds

`sign-and-notarize.sh` signs the Strata Mac app (a `.dmg` or a `.app`) and the Mac Terminal archive with the Developer ID Application
certificate, sends them to Apple's notary service, staples the ticket where that is possible, and checks the result. After it, macOS
opens the app without the "damaged" / "unidentified developer" warning.

It has to run on the Mac that holds the certificate. It never asks for, reads or stores a password or a key: `codesign` and
`notarytool` take them from the Keychain. The only names in it are the certificate (`Developer ID Application: Niko Veit (79LR4R7RSC)`)
and the notarytool profile (`strata-notary`); `SIGN_ID` and `NOTARY_PROFILE` override them.

## What to type (once per build)

1. Put these on the Mac in one folder: `sign-and-notarize.sh`, `entitlements-node.plist` (they must stay together), and the file(s) to sign,
   for example `Strata-tester-mac.dmg` and `Strata-Terminal-Mac.zip`.
2. Open Terminal and go to that folder:  `cd ~/Downloads/sign`
3. Check the two things the script needs (both should answer without an error):

       security find-identity -v -p codesigning        # lists "Developer ID Application: Niko Veit (79LR4R7RSC)"
       xcrun notarytool history --keychain-profile strata-notary

4. Run it (the app and the Terminal archive together; either part can be left out):

       bash sign-and-notarize.sh Strata-tester-mac.dmg --terminal Strata-Terminal-Mac.zip

   macOS may ask once whether `codesign` may use the key in your Keychain: type your Mac login password and choose "Always Allow".
   Then wait: Apple's check takes a few minutes per file (up to about 15 minutes each; the script waits and prints when it is done).
5. Read the summary at the end. Every line is `OK`; the finished files are in the `signed` folder next to the input, with their sha256
   (also in `signed/SHA256SUMS.txt`) and the full log (`signed/sign-and-notarize.log`).
6. If a line says `FAIL`, the run stopped there. If Apple rejected a file, the script prints Apple's log and saves it as
   `signed/notary-log-*.json`. Send us that file (it contains file names and reasons, no secrets).

A trial run that signs ad-hoc and sends nothing to Apple: add `--dry-run` (no certificate needed).

## The in-app update file (release builds only)

The app updates itself on a Mac from `Strata.app.tar.gz`, not from the dmg. The one `tauri build` makes holds the app before this script signs it, so it must
not be published. For a release build add `--updater-archive`:

       bash sign-and-notarize.sh Strata.dmg --updater-archive

After the app is signed, notarised and stapled, the script packs it into `signed/Strata.app.tar.gz` (the `.app` folder at the root of a gzip tar, like Tauri's own),
unpacks it again in a scratch folder and runs `codesign --verify --deep --strict`, `stapler validate` and `spctl` on the unpacked copy. The archive is **not signed with the Tauri
updater key** here: that key never goes near the Mac, and the Mac needs no node. Copy `Strata.app.tar.gz` to the release PC and, there:

       npx tauri signer sign -f <path to tauri-updater.key> Strata.app.tar.gz
       python packaging/feed_tool.py set-app-platform --app-json RELEASE/tauri-latest.json --target darwin-aarch64 --version V --archive Strata.app.tar.gz --sig Strata.app.tar.gz.sig

`set-app-platform` refuses a signature that does not verify for these exact bytes, then records the url, signature, SHA-256 and size. Full release order: `packaging/UPDATES.md`, "Releasing".

## What it does

1. Copies the `.app` out of the dmg into a temporary folder (the original is never changed).
2. Signs every Mach-O inside it, deepest first (node, python, every `.dylib`, `.so` and `.node`, the engine and its picture reader `strata-vision`, `rg`, `spawn-helper`,
   the app's own program): hardened runtime and a secure timestamp. Then signs the `.app`.
3. Rewrites `engine-component.json` for the engine it just signed. Control checks every engine file against the SHA-256 in that file, and
   signing changes the engine binary, so without this step the signed app would refuse its own engine.
4. `codesign --verify --deep --strict`, `spctl -a -vvv`.
5. Runs the signed copy before Apple sees it: the app's own `--self-test` (private node, host, UI, Control), every native addon loaded in
   the signed node, the bundled python (numpy, PIL, ctypes callbacks), the engine's and the picture reader's `--version` (no model is loaded), and Control's own
   check of the engine manifest. A signing mistake shows up here, not on a tester's Mac.
6. Notarises the `.app`, staples it, rebuilds the dmg (the app, an Applications link, the volume icon, as in
   `apps/desktop/scripts/make-dmg.sh`), signs the dmg, notarises and staples it, then `spctl -a -t open --context context:primary-signature -vv`.
7. Terminal archive: signs every Mach-O inside, packs it again in the format it came in (`.zip` or `.tar.gz`), and notarises it.
   A finished file is written to `signed/` only after every step for it has passed.

## Entitlements (each one is a hole in the hardened runtime)

Only the bundled Node gets one: `com.apple.security.cs.allow-jit` (`entitlements-node.plist`, with the measurements). Python, the engine,
the app shell and all libraries get none. `disable-library-validation` is not used: library validation only refuses a library signed
by somebody else, and every library in these bundles is re-signed with the one identity. If a tester ever reports a "different Team IDs"
error for a library that is not in the bundle (for example an addon an MCP server installs and loads with this node), add that key to
`entitlements-node.plist` and run the script again.

The engine components the app downloads later (`strata update`) are unpacked by Control without the download mark, run as their own
processes, and are not hardened by the app's runtime (a child process is not bound by its parent's hardened runtime), so unsigned
components keep working. The engine copied out of the app's bundle keeps our signature; it needs no entitlement (Metal compiles the
kernel sources through the system compiler service, tested under the hardened runtime).

## The Terminal zip

Apple cannot staple a ticket to bare command-line programs in a zip. The zip is still notarised (Apple records the programs by hash),
and macOS looks that up online the first time a browser-downloaded copy is opened. Installs done with `curl ... | sh` are not
quarantined at all, so this mainly helps zips downloaded in a browser.

Signing changes the bytes of the zip and of every program inside, so:

* a checksum list or signed feed (`strata-feed.json`, `*.manifest.json`) that names the old zip or archive must be rebuilt from the new one;
* an archive that holds a *feed* (the tester Terminal zip: scripts plus `feed/*.tar.gz`) has no program at its top level. By default
  the nested `.tar.gz` files are left alone (they are unpacked by command-line tools and carry no download mark). `--sign-nested`
  signs inside them and packs them again, but then the feed's `sha256` entries and its signature (`strata-feed.json.sig`, made with the
  release key) no longer match and the feed has to be regenerated and re-signed.
