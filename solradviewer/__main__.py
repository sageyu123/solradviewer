"""Allow ``python -m solradviewer`` to launch the application."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
