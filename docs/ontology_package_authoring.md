# Ontology Package Authoring

Kogwistar ontology packages are versioned, data-only JSON documents. They
describe classes, scalar properties, relations, and role-shaped edges without
registering code or granting graph mutation authority.

## Validate A Package

The official schema is available from the CLI:

```text
kogwistar ontology schema
kogwistar ontology validate examples/ontology/communication.json
kogwistar ontology validate examples/ontology/communication.json --json
kogwistar ontology validate examples/ontology/email.json --compose-with examples/ontology/communication.json
```

Validation is read-only. Exit code `0` means valid; `2` means malformed JSON or
schema/semantic shape errors; `3` means a digest, identity, or composition
failure.

## Package Shape

Each bundle has a `manifest` and at least one descriptor:

```json
{
  "manifest": {
    "schema_version": 1,
    "ontology_id": "communication",
    "version": "1.0.0",
    "title": "Communication",
    "summary": "Message and exchange vocabulary.",
    "content_sha256": "<64 lowercase hex characters>",
    "imports": [],
    "extensions": {}
  },
  "descriptors": [
    {
      "kind": "ontology_property",
      "descriptor_id": "subject",
      "name": "Subject",
      "value_kind": "string",
      "min_count": 0,
      "max_count": 1
    }
  ]
}
```

The digest is SHA-256 over canonical JSON with `manifest.content_sha256`
removed. Object keys, imports, and descriptors are canonicalized, so harmless
JSON reordering does not change the package identity.

## Descriptor Kinds

- `ontology_property`: a scalar value type (`string`, `integer`, `number`,
  `boolean`, `timestamp`, `logical_ref`, or `json`) and cardinality.
- `ontology_class`: a named class with local or qualified property IDs.
- `ontology_relation`: source and target class references.
- `ontology_edge_shape`: an existing graph edge shape with named roles. A role
  targets either a `node` or an `edge` and may restrict allowed class IDs.

IDs are ASCII identifiers beginning with a letter. References may be local
(`Message`) or qualified (`communication:Message`). Cardinalities are bounded
when `max_count` is present; `max_count: null` means unbounded above.

## Imports And Extensions

Imports pin the exact `ontology_id`, semantic version, and content digest of a
dependency. Composition fails closed for missing exact imports, cycles, digest
mismatches, duplicate qualified IDs, and incompatible definitions. Composition
does not depend on input order.

Extensions are optional JSON primitives, arrays, or objects under namespaced
keys such as `vendor/key`. They cannot contain executable callbacks, import
paths, shell commands, network loading instructions, or plugin code.

The checked-in schema artifact is
`contracts/ontology/ontology_package.v1.schema.json`; regenerate it with:

```text
python scripts/export_ontology_schema.py
```
