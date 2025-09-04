"""Библиотека для генетической стратификации данных."""
from .genetic_stratifier import (
    GeneticStratifier,
    Stratification,
    stratify_with_inheritance,
)
from .binning import bin_numeric

__all__ = [
    "GeneticStratifier",
    "Stratification",
    "stratify_with_inheritance",
    "bin_numeric",
]
