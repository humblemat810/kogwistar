# Chat Client Debugging Checklist

This note captures the failure modes we hit while wiring the FastHTML/HTMX chat client against the server.

## Common Pitfalls

- `localhost` is not always the same backend as `127.0.0.1` on Windows.
- Port `28110` is shared by multiple possible listeners in this repo setup, including Docker and the local VS Code-launched server.
- Port `5173` is the frontend dev server URL. The backend expects the UI to exist there, but compose does not start it.
- `DEV_AUTH_NS` controls which namespaces the dev JWT can access. If it is `docs`, the dev user will not reach conversation/workflow routes.
- `/auth/dev-token` expects a JSON body. Empty or form-encoded requests fail before namespace validation.
- `AUTH_MODE=dev` changes auth behavior. If you are debugging login, confirm whether the server was started in `dev` or `oidc`.
- `/api/document.upsert_tree` requires the document to already exist.
- `persist_document_graph_extraction(...)` validates node and edge spans against the stored document content.
- `persist_document_graph_extraction(...)` also resolves edge endpoints, so endpoint IDs must be structurally valid.
- `curl` in PowerShell may not be the `curl.exe` binary you think it is.
- `curl http://localhost:28110/...` and `curl.exe http://127.0.0.1:28110/...` can hit different listeners.
- The request logger is separate from the app logic. A silent console does not mean the request failed.

## Q&A

- Q: Why do I get `Not Found` but no log line?
  - A: You may be hitting a different listener on `localhost`, or the request may be going through a path handled outside the logger you are watching. Use `curl.exe` and prefer `127.0.0.1`.

- Q: Why does the dev account only see `docs`?
  - A: The VS Code dev-auth profile was previously set to `DEV_AUTH_NS=docs`. Set it to `docs,conversation,workflow,wisdom` if you want full access.

- Q: Why does `/auth/dev-token` fail before it even checks `ns`?
  - A: The endpoint parses JSON first. If the client sends no body or non-JSON form data, you get a JSON decode failure instead of an auth validation error.

- Q: Why does `/api/document.upsert_tree` fail on my seed payload?
  - A: The document must exist first, and the spans must match the document content. A mismatched excerpt or missing document will fail validation.

- Q: Why does a debug workflow appear to work in tests but not in the live server?
  - A: Test fixtures often use fake backends and direct engine setup. The live server also needs the correct auth namespace, the document row, and a valid backend route.

- Q: Why does the conversation client get `401` on chat routes?
  - A: The token probably lacks the required namespace or role. Check `/api/auth/me` and verify the `ns` claim.

- Q: Why do SSE events sometimes disappear?
  - A: The stream may be connected to the wrong port or the wrong process, or the client may be using `localhost` while the server is bound on `127.0.0.1`.

## Setup Checklist

- Start the backend on a known port and verify it is the process you intend to test.
- Start or point the frontend at `http://localhost:5173/` if you are using the default UI URL.
- Confirm the VS Code launch profile or compose profile sets the intended auth mode.
- Confirm `DEV_AUTH_NS` contains the namespaces the client needs.
- Confirm `/auth/dev-token` is being called with `Content-Type: application/json`.
- Confirm `UI_URL` matches the actual browser app URL.
- Confirm `OIDC_REDIRECT_URI` points back to the backend when using OIDC.
- Mint a token and inspect `/api/auth/me` before testing chat routes.
- Verify `/api/conversations` works before testing answer submission.
- Create the document before calling `/api/document.upsert_tree`.
- Keep span excerpts and offsets aligned with the document content.
- Use `curl.exe -v http://127.0.0.1:28110/...` when checking the local debug server.

## Recommended Dev Auth Settings

Use these values for the VS Code debug profile when you want full local access:

```json
{
  "AUTH_MODE": "dev",
  "DEV_AUTH_EMAIL": "dev@example.com",
  "DEV_AUTH_SUBJECT": "dev",
  "DEV_AUTH_NAME": "Dev User",
  "DEV_AUTH_ROLE": "rw",
  "DEV_AUTH_NS": "docs,conversation,workflow,wisdom",
  "UI_URL": "http://127.0.0.1:5173/"
}
```

## Fast Checks

- `GET /api/auth/me` should show the expected `role` and `ns`.
- `GET /health` should return the server you think you started.
- `GET /api/conversations` should return data once you have a valid token.
- `POST /api/document` should succeed before any tree upsert.
- `POST /api/document.upsert_tree` should only use spans that match the document text.
