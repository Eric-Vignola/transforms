"""
Cross-platform Numba cache configuration.

Redirects Numba's compiled cache files away from ``__pycache__/`` directories
in the source tree to a platform-appropriate hidden location, namespaced by
the installed Numba version to avoid stale-cache issues on upgrades.

Call :func:`configure_numba_cache` **before** any ``@njit(cache=True)``
decorated module is imported.

Cache locations
---------------
- **Windows** : ``%LOCALAPPDATA%\\numba_cache\\<version>\\``
- **macOS**   : ``~/Library/Caches/numba/<version>/``
- **Linux**   : ``$XDG_CACHE_HOME/numba/<version>/``  (defaults to ``~/.cache``)

The ``NUMBA_CACHE_DIR`` environment variable, if set, takes precedence and
disables the automatic redirect.
"""

import os
import sys

_configured: bool = False

# Bump this when making cross-function changes that Numba's source
# stamp might miss (e.g. modifying a callee without touching the caller).
# This will force Numba to recompile all functions that use the cache.
# Also this is not expected to happen often, if ever, and added purely as a
# safeguard.
_CACHE_VERSION: int = 1


def configure_numba_cache() -> None:
    """Redirect Numba's file cache to a platform-appropriate, version-namespaced location."""
    global _configured
    if _configured:
        return
    _configured = True

    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    from numba import __version__ as numba_version, config

    version_key = f"{numba_version}_v{_CACHE_VERSION}"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        config.CACHE_DIR = os.path.join(base, "numba_cache", version_key)
    elif sys.platform == "darwin":
        config.CACHE_DIR = os.path.join(
            os.path.expanduser("~"), "Library", "Caches", "numba", version_key
        )
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        config.CACHE_DIR = os.path.join(base, "numba", version_key)

configure_numba_cache()
