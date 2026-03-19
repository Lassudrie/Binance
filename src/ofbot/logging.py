from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(verbose: bool = False) -> None:
    """Configure structured console logging for paper engine runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True)],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

