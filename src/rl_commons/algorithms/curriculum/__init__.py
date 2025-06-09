"""Curriculum Learning algorithms for progressive task difficulty

Module: __init__.py
"""

from .base import (
    CurriculumScheduler,
    Task,
    TaskDifficulty,
    PerformanceTracker
)
from .automatic import AutomaticCurriculum
from .progressive import ProgressiveCurriculum
from .adaptive import AdaptiveCurriculum
from .meta_curriculum import MetaCurriculum

__all__ = [
    'CurriculumScheduler',
    'Task',
    'TaskDifficulty',
    'PerformanceTracker',
    'AutomaticCurriculum',
    'ProgressiveCurriculum',
    'AdaptiveCurriculum',
    'MetaCurriculum'
]