# 05 · Serving

**Route** `#/serving` · **File** `src/screens/Serving.tsx` (329 lines)

The screen for people wiring Strata into something else. It is the most
"developer" surface in the app, which makes it the one most likely to be skimmed
by someone who does not read API documentation for fun.

## What it is today

- **Local API card** — base URL, OpenAI-compatible and Anthropic-compatible
  endpoint chips, a copy button, the model id, and a "Serving · bind locally"
  pill.
- **Access card** — toggles for other devices and network access, an API key row
  with reveal/copy, and a plain-language reassurance line.
- **Requests table** — time, client, endpoint, tokens in/out, duration, state.
  A row opens a detail drawer.
- **Metrics** — running, waiting, and a few counters.
- **Empty state** for a fresh install.

## Where it falls short

1. ~~The endpoints are shown as chips, not as usable snippets.~~ **Fixed.** The
   endpoint rows printed a *path fragment* in muted mono with no way to take it,
   so wiring a client meant reading `/chat/completions` off the screen and typing
   the host yourself. Each row now carries its full URL and its own copy button.
   (A base-URL copy button did already exist — the plan over-claimed there.)
2. ~~No example request.~~ **Fixed.** A `curl` block, switched between OpenAI and
   Anthropic formats, with a copy button. Verified by *running* the copied
   command against the host: it returns real SSE completion chunks, and the JSON
   body parses.
3. **Security state is ambiguous.** "Other devices" and "network access" toggles
   exist beside an API key, but nothing states plainly whether the server is
   reachable from outside this machine, which is the actual question.
4. **The requests table has no filtering or sorting.** A hundred rows, no way to
   find the failed one — despite the state column having a colour.
5. **Failures are indistinguishable from successes at a glance.** The state pill
   is the only signal, and it is small.
6. **The detail drawer is a dead end.** It shows a request's facts and offers
   Cancel; it does not offer "copy as curl", which is what you want when
   debugging a call the model answered badly.
7. **The table does not say when it is empty because nothing has been asked** vs
   when it is filtered to nothing.

## The plan

### Substance — make it pasteable
1. **Replace the endpoint chips with a copyable base URL block** and a
   one-line example request beneath it, switched by a small OpenAI /
   Anthropic segmented control. One block, one copy button, no assembly.
2. **Add "Connect a client" with two or three concrete recipes** — an editor,
   a CLI, a script — each copyable. This is the screen's real job.
3. **State the exposure plainly**: `Only this computer can reach it` or
   `Anyone on your network can reach it`, as a sentence with a status dot, above
   the toggles rather than implied by them.
4. **Make the API key state visible at rest** — present / not set — rather than
   requiring the reveal toggle to find out.

### Hierarchy
5. **Order by intent, not by implementation**: *Connect to it* → *Who can reach
   it* → *What it has been doing*. Today the layout is API, access, table, which
   is close but reads as three equal cards.

### Fluidity
6. **Filter and sort the request table**, with the state column as a filter
   rather than only a colour. Failed-only is the view people actually want.
7. **Row state gets an icon as well as a colour**, so it survives greyscale.
8. **The detail drawer gains "Copy as cURL"** and the full request/response
   payload, collapsed.

## Motion

| Interaction | Spec |
|---|---|
| Endpoint protocol switch | Panel cross-fade only, `--st-t-control`; never both visible |
| Copy confirmation | Check icon swap + a toast; the button does not move |
| Request row → drawer | Existing drawer slide, `--st-t-local`; no shared-element morph — the sheet's own transform owns the motion |
| New row arriving | `st-enter`, `--st-t-local`, 6 px, capped at the newest 6 rows |

## Verification

- **Shipped and verified**: every endpoint row copies a full URL, the request
  block copies a `curl` that executes, and the Anthropic format differs from the
  OpenAI one (`max_tokens`, `/messages`).
- Assert the exposure sentence matches the toggle state in all four combinations.
- Assert request rows expose state as both icon and colour.
- `flow serving`: toggle each access switch, filter to failures, open a drawer.
