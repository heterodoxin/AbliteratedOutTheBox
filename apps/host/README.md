# Strata app source

Start with [the complete developer handoff](../handoff.md) and [the runbook](../docs/RUNBOOK.md).

```sh
npm ci
npm run preview:ui
```

The built UI is included. To rebuild after source edits, run `npm run build`. `preview:ui` uses isolated simulated data; `npm run dev` connects the normal host.

The previous project README is retained in [docs/README-historical.md](docs/README-historical.md). It contains older implementation status and is historical reference.
