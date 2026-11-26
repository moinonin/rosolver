"""Allow `python -m rosolver.mcp` execution."""

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
