# ADR-017: Sigma CDC Hypergraph Viewer

Status: Accepted

Date: 2026-08-12

## Context

Kogwistar is a generic directed hypergraph substrate. An edge may have multiple
source nodes, multiple target nodes, source edges, and target edges. Existing D3
and Cytoscape viewers are useful compatibility surfaces, but a second CDC viewer
is needed for larger, more interactive graphs and for human visual evaluation.

Treating every hyperedge as a set of binary links loses edge identity, endpoint
roles, meta-edge structure, and provenance. That loss is especially dangerous in
a live CDC view because the picture can appear valid while no longer representing
the event stream faithfully.

## Decision

Add Sigma.js as a parallel viewer. Do not replace or change the public behavior
of the D3 and Cytoscape viewers.

The browser keeps one canonical raw model:

- entity nodes keyed by stable entity ID;
- hyperedges keyed by stable edge ID;
- `source_ids`, `target_ids`, `source_edge_ids`, and `target_edge_ids` preserved;
- CDC sequence, timestamp, graph type, run ID, and step ID retained;
- placeholders allowed when an edge arrives before an endpoint.

All render modes are projections from that raw model:

1. **Reify** is authoritative and lossless. Every hyperedge is a visible node.
   Incidence links retain `src-node`, `tgt-node`, `src-edge`, or `tgt-edge` role.
2. **Compact** remains hypergraph-aware. Binary edges may render directly, while
   true hyperedges and meta-edges remain reified. The UI identifies it as a mixed
   projection.
3. **Projection** expands node sources to node targets as derived binary links.
   It is explicitly lossy. Each derived link retains its originating hyperedge ID
   for inspection and return to reified mode. Meta-edge-only relationships cannot
   be represented as ordinary binary links and remain reified.

The raw model, not Graphology, is authoritative. Graphology is rebuilt or updated
from the selected projection; Sigma is the renderer and interaction surface.

CDC normalization accepts canonical payloads and established legacy aliases:
`source_ids`/`source_id`/`source`, `target_ids`/`target_id`/`target`, and equivalent
edge-ID singular/plural forms. It never silently converts an edge endpoint into a
node endpoint.

The existing change bridge exposes human test pages:

- a Sigma viewer playground connected to `/changes/ws`;
- a CDC event forge that sends realistic events to `/ingest`;
- scenario controls for nodes, multi-source/multi-target hyperedges, meta-edges,
  updates, removals, replay, and bursts.

These pages exercise the same ingest, durable oplog, replay, filtering, and live
tail path used by normal CDC consumers.

## Interaction and visual contract

- Entities and hyperedges must be visually distinct without relying on color
  alone. Hyperedges use diamond geometry on a Sigma-managed rendering layer.
- Source and target incidence are directionally and chromatically distinct.
  Each directed segment also follows the established D3 value-gradient
  contract: high-value color at its source falls to low-value color at its
  target, so direction remains legible when arrowheads are obscured by density.
- Selection reveals one-hop incidence, dims unrelated structure, and opens a
  readable inspector containing raw payload and provenance.
- Search, graph-type filtering, render-mode switching, zoom, reset, dragging,
  and CDC pause/resume must have visible state.
- Mode changes and CDC updates preserve stable node positions where practical.
- The viewport reports connection state, last sequence, raw node/edge counts,
  visible counts, and whether the current mode is lossless or derived.

## Verification

Compilation and syntax checks are necessary but insufficient. Acceptance requires
Playwright-driven observation of the running pages:

- create events in the forge and see them appear in the viewer;
- exercise selection, inspection, search, filters, dragging, zoom, and each mode;
- verify replay and live updates, including endpoint-before-node ordering;
- capture screenshots for reified, compact, projected, selected, and updated
  states and inspect them for overlap, clipping, contrast, hierarchy, and feedback;
- record browser console and page errors.

Semantic/unit tests cover projection rules and CDC normalization. Existing D3 and
Cytoscape tests remain regression gates. Native code is not changed by this ADR;
if later native work is required, both Cargo and Python parity tests are mandatory.

## Consequences

Sigma and Graphology add a browser dependency. The debug bridge serves pinned,
vendored Sigma.js 2.4.0 and Graphology 0.25.4 assets, so browser acceptance does
not depend on external network state. Generated standalone HTML bundles retain
pinned CDN URLs. This changes neither raw data nor projection contracts.

The projection view can be visually simpler but is never presented as the source
hypergraph. Reification remains the default and the reference for debugging.
