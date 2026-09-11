# Sigma CDC Hypergraph Lab

The lab is a human-facing companion to the Kogwistar CDC bridge. It exercises
the durable ingest, oplog replay, graph-type filter, and WebSocket live-tail path.

Start it from the repository root:

```powershell
.venv\Scripts\python.exe -m kogwistar.cdc.change_bridge `
  --host 127.0.0.1 `
  --port 8787 `
  --oplog-file .cdc_debug/data/cdc_oplog.jsonl
```

Open `http://127.0.0.1:8787/debug` and use both links:

- `/debug/sigma?stream=knowledge` is the interactive Sigma viewer.
- `/debug/forge` creates node, hyperedge, meta-edge, update, and remove events.

The forge's **Generic hypergraph** scenario intentionally contains one ordinary
binary edge, two many-endpoint hyperedges, and two meta-edges. This makes the
three viewer modes visibly different:

- **Reify** is the lossless reference. Every edge is a diamond and every
  incidence role is visible.
- **Compact** collapses only the ordinary binary edge. True hyperedges and
  meta-edges remain reified.
- **Projection** expands node-to-node incidences into dashed derived links.
  Edge-to-edge structure remains reified because it cannot be represented as an
  ordinary binary graph without losing its endpoint kind.

All directed segments fade from a brighter source side to a darker target side,
matching the D3 viewer's direction cue. Hue still distinguishes source
incidence from target incidence; arrowheads remain a redundant second cue.

The other scenarios test endpoint-before-node ordering and rapid
create/update/remove replay. Use **Pause CDC**, send a scenario, and resume to
observe catch-up from the viewer's last sequence.

## Visual acceptance

Install the repository's browser extra and Chromium once:

```powershell
.venv\Scripts\pip.exe install -e ".[browser]"
.venv\Scripts\python.exe -m playwright install chromium
```

Run the interaction suite:

```powershell
.venv\Scripts\python.exe -m pytest tests/cdc/test_sigma_cdc_playwright.py -q
```

Screenshots are written to `.tmp_sigma_visuals` by default. Set
`KOGWISTAR_VISUAL_ARTIFACTS` to retain them elsewhere. Passing assertions alone
is not visual acceptance: inspect the reify, compact, projection, selected,
endpoint-ordering, and replay screenshots for clipping, overlap, contrast,
direction cues, and understandable feedback.

The bridge serves pinned, vendored Sigma.js 2.4.0 and Graphology 0.25.4 assets,
so the live lab has no CDN dependency. Generated standalone bundles keep pinned
CDN URLs because a single HTML file cannot resolve package-relative assets.
Third-party MIT notices ship with the templates.
