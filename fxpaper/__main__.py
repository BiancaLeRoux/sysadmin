"""Entry point so the simulator can be run as ``python -m fxpaper``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
