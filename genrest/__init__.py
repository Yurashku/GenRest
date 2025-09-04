"""Библиотека для генетической стратификации данных."""
from .genetic_stratifier import GeneticStratifier, Stratification
from .binning import bin_numeric

__all__ = ["GeneticStratifier", "Stratification", "bin_numeric"]
