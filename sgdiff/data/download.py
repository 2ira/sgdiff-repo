"""Network download with mirror fallback.

The reproduction environment often cannot reach ``github.com`` directly, while
``raw.githubusercontent.com`` and generic mirrors are reachable.  Every download
therefore tries the URL as-is first and then a list of mirrors.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from ..utils.io import ensure_dir
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

USER_AGENT = "sgdiff-repro/0.1 (+https://github.com/hzlihang99/SGDiff)"

#: Prefixes prepended to a URL when the direct request fails.
DEFAULT_MIRRORS: tuple[str, ...] = (
    "",
    "https://ghproxy.net/",
)

#: Used for ``github.com`` pages (HTML), which some mirrors do not proxy.
PAGE_MIRRORS: tuple[str, ...] = (
    "https://r.jina.ai/",
)


class DownloadError(RuntimeError):
    """Raised when every mirror failed for a resource."""


def download_file(
    url: str,
    dest: str | Path,
    mirrors: Sequence[str] = DEFAULT_MIRRORS,
    timeout: int = 60,
    retries: int = 2,
    min_bytes: int = 1,
) -> Path:
    """Download ``url`` into ``dest``, trying each mirror prefix in order.

    Existing non-empty files are reused, which makes dataset preparation
    idempotent and offline-friendly.
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        LOGGER.debug("cache hit: %s", dest)
        return dest

    errors: list[str] = []
    for prefix in mirrors:
        candidate = f"{prefix}{url}" if prefix else url
        try:
            _stream_to_file(candidate, dest, timeout=timeout, retries=retries, min_bytes=min_bytes)
            LOGGER.info("downloaded %s -> %s", candidate, dest.name)
            return dest
        except Exception as exc:  # noqa: BLE001 - a mirror may fail in many ways
            errors.append(f"{candidate}: {exc}")
            LOGGER.warning("download failed via %s (%s)", candidate, exc)

    raise DownloadError(
        "unable to fetch {url}\ntried:\n  ".format(url=url) + "\n  ".join(errors)
    )


def _stream_to_file(url: str, dest: Path, timeout: int, retries: int, min_bytes: int) -> None:
    last_error: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as fh:
                shutil.copyfileobj(response, fh)
            if tmp.stat().st_size < min_bytes:
                raise IOError(f"downloaded file is too small ({tmp.stat().st_size} bytes)")
            tmp.replace(dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
    raise last_error if last_error else IOError("unknown download failure")


def download_many(items: Iterable, **kwargs) -> list[Path]:
    """Download several ``(url, destination)`` pairs in order."""
    return [download_file(url, dest, **kwargs) for url, dest in items]
