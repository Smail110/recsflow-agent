"""Каталог RecAgent: генератор синтетических объектов и профили пользователей.

Оба модуля детерминированы по seed и не читают сеть. Калибровка — в calibration.py,
банки слов для названий — в words.py.
"""

from __future__ import annotations

from .calibration import CATALOG_SIZE
from .generator import catalog_sha256, catalog_stats, generate_catalog, title_pool_size

__all__ = ["CATALOG_SIZE", "catalog_sha256", "catalog_stats", "generate_catalog", "title_pool_size"]
