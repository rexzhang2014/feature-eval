# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "pandas",)
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


# from featureval.metrics import *
from featureval.redrel import (
    Redundancy, 
    Relevance
)

from featureval.selection import (
    mRMR # Only provide this default algorithm for convinience.
)

from featureval.preprocess import(
    categories_to_integer,
    zscore_normalize,
    detect_variable_type,
    weight_of_evidence,
)

from featureval import (
    metrics,
    preprocess,
    selection,
    redrel
)