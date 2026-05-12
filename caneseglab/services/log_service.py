from __future__ import annotations

import logging
from typing import Any


class LogService:
    """全局日志服务。"""

    def __init__(self, name: str = "caneseglab", level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(name)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self._logger.addHandler(handler)

        self._logger.setLevel(level)
        self._logger.propagate = False

    def set_level(self, level: str | int) -> None:
        self._logger.setLevel(self._normalize_level(level))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(message, *args, **kwargs)

    @staticmethod
    def _normalize_level(level: str | int) -> int:
        if isinstance(level, int):
            return level

        level_name = level.strip().upper()
        mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        if level_name not in mapping:
            raise ValueError(f"Unsupported log level: {level}")
        return mapping[level_name]


logger = LogService()
