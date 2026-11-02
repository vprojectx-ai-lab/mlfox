from .ingestion import load_dataset
from .profiler import DatasetProfile, profile_dataset, detect_task, is_discrete_numeric_multiclass
from .splitter import DatasetSplit, split_dataset
from .feature_selection import (
    FeatureSelectionIntelligence,
    FeatureSelectionReport
)
from .column_intelligence import (
    ColumnIntelligenceReport,
    analyze_columns
)

from .missing_intelligence import (
    MissingValueIntelligence,
    MissingValueReport
)

from .feature_engineering import (
    FeatureEngineeringIntelligence,
    FeatureEngineeringReport
)

from .encoding import (
    EncodingScalingIntelligence,
    EncodingReport
)
from .pipeline import FoldAwarePreprocessor
from .model_selection import ModelSelectionEngine
from .tuning import HyperparameterTuner
from .mlfox import MLFOX, mlfox
from .persistence import load_model
import sys as _sys
from types import ModuleType as _ModuleType

__all__ = [
    "load_dataset",
    "DatasetProfile",
    "profile_dataset",
    "detect_task",
    "is_discrete_numeric_multiclass",
    "DatasetSplit",
    "split_dataset",
    "ColumnIntelligenceReport",
    "analyze_columns",
    "MissingValueIntelligence",
    "MissingValueReport",
    "FeatureEngineeringIntelligence",
    "FeatureEngineeringReport",
    "EncodingScalingIntelligence",
    "EncodingReport",
    "FeatureSelectionIntelligence",
    "FeatureSelectionReport"
    , "FoldAwarePreprocessor", "ModelSelectionEngine", "HyperparameterTuner", "MLFOX", "mlfox", "load_model"
]


class _CallableMLFOXModule(_ModuleType):
    """Allow ``import mlfox`` to provide the callable ``mlfox(...)`` API."""

    def __call__(self, *args, **kwargs):
        return self.mlfox(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableMLFOXModule
