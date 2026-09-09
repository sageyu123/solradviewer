"""Command-line entry point for the SolRadViewer web application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn


def _port(value: str) -> int:
    """Parse a TCP port and reject values outside the usable range."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: Sequence[str] | None = None) -> int:
    """Run the packaged SolRadViewer backend."""
    parser = argparse.ArgumentParser(
        prog="solradviewer",
        description="Run the SolRadViewer web application.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        default=8010,
        type=_port,
        help="TCP port to bind (default: 8010)",
    )
    args = parser.parse_args(argv)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"SolRadViewer: http://{display_host}:{args.port}", flush=True)
    uvicorn.run("solradviewer.backend.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by python -m
    raise SystemExit(main())
