"""Console launcher for the Rust server embedded in the native wheel."""
from __future__ import annotations

from ._rust_bridge import RustExtensionUnavailableError, _load_extension, server_implementation_mode


def main() -> None:
    if server_implementation_mode() != "rust":
        raise SystemExit(
            "kogwistar-rust-server requires KOGWISTAR_IMPL_SERVER=rust; "
            "Python server ownership remains selected"
        )
    try:
        extension = _load_extension()
    except RustExtensionUnavailableError as error:
        raise SystemExit(str(error)) from error
    run = getattr(extension, "api_run_server", None)
    if run is None:
        raise SystemExit("installed native extension does not include Rust server support")
    run()


if __name__ == "__main__":
    main()
