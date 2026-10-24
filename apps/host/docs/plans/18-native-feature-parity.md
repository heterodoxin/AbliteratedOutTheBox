# Native feature parity — 14 October 2026

Consumer flow: choose a compatible model, optionally read pictures, download/load, chat. Native installation uses Strata's own installer, including shared shards, MTP, packing, vision and resumable downloads. Expert mode exposes device placement, native configuration and parallel instances. Settings > Engine contains tools, measurements and diagnostics. No model downloads, native updates or calibration are run merely by opening the UI.

Implementation checklist:

- [x] Native control adapter: official MCP installer/catalog, isolated native engine instances, measured telemetry, jobs and cancellation.
- [x] Model catalog and first-run choices from installed Strata's tables, with honest hardware fit and existing-file reuse.
- [x] Image understanding: optional encoder setup and image payloads; generation stays a separate option.
- [x] Responses gateway and three protocol connection templates.
- [x] Native parameter editor with validation, original configuration preservation and backups.
- [x] MCP manager and chat tool opt-in.
- [x] Download/startup recovery and engine logs.
- [x] Measured benchmark, calibration, native updates and engine rollback.
- [x] All-model Monitor speed table and per-instance details.
- [x] Focused fixture tests, TypeScript/build, live UI evidence and handover.

The design host remains available for existing preview conversations. Native operations are explicit actions and use a configured local Strata checkout. Real hardware speed and memory allocation are shown only when reported. Backend absence produces a clear setup path, never a simulated successful operation.

Verification: 149 tests passed; TypeScript and production build passed. Browser checks cover consumer loading, expert devices, multiple-instance Monitor, MCP saving and benchmark results. See `docs/native-engine-handover.md` for contracts and actual GPU verification limits. The official native checkout remained unmodified.
