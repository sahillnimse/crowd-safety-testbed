"""Entry point: `python -m webapp` starts the UI server."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Crowd Safety Testbed web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (development).")
    args = parser.parse_args()

    import uvicorn
    print(f"\n  Crowd Safety Testbed  ->  http://{args.host}:{args.port}\n")
    uvicorn.run("webapp.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
