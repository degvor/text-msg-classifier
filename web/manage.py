#!/usr/bin/env python
import os
import sys
from pathlib import Path


def main() -> None:
    # Load .env for local runs. In containers the values arrive via
    # docker-compose `env_file`, and load_dotenv is a no-op when the file is
    # absent, so this is safe in both.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:  # pragma: no cover
        pass

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
