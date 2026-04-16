"""Fallback downloader for stable_datasets when requests gets blocked."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock
from requests import HTTPError

import stable_datasets.utils as stable_utils


_ORIGINAL_DOWNLOAD = stable_utils.download


def _wget_download(
    url: str,
    dest_folder: str | Path | None = None,
    progress_bar: bool = True,
    _progress_dict=None,
    _task_id=None,
) -> Path:
    if dest_folder is None:
        dest_folder = stable_utils._default_dest_folder()
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    p = Path(filename)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    dest = dest_folder / f"{p.stem}.{h}{p.suffix}"
    lock = dest.with_suffix(dest.suffix + ".lock")
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    with FileLock(lock):
        if dest.exists():
            return dest

        cmd = ["wget", "-O", str(tmp), url]
        if not progress_bar:
            cmd.insert(1, "-q")

        try:
            logging.info("Falling back to wget for %s", url)
            subprocess.run(cmd, check=True)
            tmp.replace(dest)
            logging.info("wget download finished: %s", dest)
            return dest
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise


def patched_download(
    url: str,
    dest_folder: str | Path | None = None,
    progress_bar: bool = True,
    _progress_dict=None,
    _task_id=None,
) -> Path:
    try:
        return _ORIGINAL_DOWNLOAD(
            url,
            dest_folder=dest_folder,
            progress_bar=progress_bar,
            _progress_dict=_progress_dict,
            _task_id=_task_id,
        )
    except HTTPError as exc:
        if exc.response is None or exc.response.status_code != 403:
            raise
        return _wget_download(
            url,
            dest_folder=dest_folder,
            progress_bar=progress_bar,
            _progress_dict=_progress_dict,
            _task_id=_task_id,
        )


def apply_patch() -> None:
    stable_utils.download = patched_download
