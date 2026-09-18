import logging
import os
import tempfile
from collections.abc import Mapping
from typing import Any

import torch

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def save_checkpoint(payload: Mapping[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location=map_location)
    return payload
