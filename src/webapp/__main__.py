"""Entry point: `python -m webapp` starts the UI server."""

import argparse
import os
import sys

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _setup_logging() -> None:
    """
    Root logger -> console AND outputs/logs/webapp.log (explicit utf-8).

    Without this the library loggers have no handler at all: model startup
    lines that tell an operator what actually ran (calibration state, zone
    resolution, vehicle/umbrella detector tier, threshold-disable reasons)
    are emitted at INFO and never appear anywhere in a webapp session, and
    WARNINGs survive only on stderr while a console happens to be attached.
    The file sink persists all of it across restarts. utf-8 is explicit so
    non-ASCII text in log messages cannot die on Windows' cp1252 console
    codec.
    """
    import logging

    log_dir = os.path.join(_PROJECT_ROOT, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(log_dir, "webapp.log"), encoding="utf-8",
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Crowd Safety Testbed web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (development).")
    args = parser.parse_args()

    # Before importing app/models: early load-time loggers must have a
    # handler to record to, not just the ones created per-request.
    _setup_logging()

    # Refuse to expose an unauthenticated server on the network.
    #
    # The API can start and cancel jobs, read every stored video, and wipe
    # outputs/ entirely. Bound to loopback that is a local tool; bound to
    # 0.0.0.0 without a token it is all of that, for anyone who can reach the
    # host. Defaulting to "insecure but convenient" is how a testbed ends up
    # deployed as-is, so the unsafe combination is refused rather than warned
    # about.
    _loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not _loopback and not os.environ.get("CROWD_API_TOKEN"):
        raise SystemExit(
            f"\nREFUSING TO START: --host {args.host} exposes this server "
            f"beyond localhost, but CROWD_API_TOKEN is not set.\n\n"
            f"  Every /api endpoint would be open, including job control and\n"
            f"  DELETE /api/outputs (which wipes all stored runs).\n\n"
            f"Set a token first:\n"
            f'  PowerShell:  $env:CROWD_API_TOKEN = "<a long random string>"\n'
            f'  bash:        export CROWD_API_TOKEN="<a long random string>"\n\n'
            f"Or bind to localhost only:  python -m webapp --host 127.0.0.1\n"
        )
    if os.environ.get("CROWD_API_TOKEN"):
        print("  API token authentication: ENABLED")

    import uvicorn
    print(f"\n  Crowd Safety Testbed  ->  http://{args.host}:{args.port}\n")
    print(f"  Logs: {os.path.join(_PROJECT_ROOT, 'outputs', 'logs', 'webapp.log')}\n")
    uvicorn.run("webapp.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
