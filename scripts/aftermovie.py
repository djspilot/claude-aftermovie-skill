#!/usr/bin/env python3
"""Back-compat shim — delegates to the installed `aftermovie` package.

The shim filename collides with the package name, so we strip this script's
directory from sys.path before importing.
"""
import os
import sys

_here = os.path.realpath(os.path.dirname(__file__))
sys.path = [p for p in sys.path if os.path.realpath(p) != _here]

from aftermovie.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
