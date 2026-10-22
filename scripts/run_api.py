"""CLI script to start the VouchRec FastAPI research backend server."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from vouchrec.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Start VouchRec Research Dashboard API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable live code reloading")
    args = parser.parse_args()

    logger.info("Starting VouchRec Research API at http://%s:%d", args.host, args.port)
    uvicorn.run(
        "vouchrec.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
