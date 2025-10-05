#!/usr/bin/env python3
"""Entry point for the formation flight PID control demonstration."""
from __future__ import annotations

from .simulation import demo_sim


def main() -> None:
    """Main entry point for the formation flight demonstration."""
    demo_sim()


if __name__ == "__main__":
    main()
