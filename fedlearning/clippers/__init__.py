from .clipbyval import ClipByValue
from .clipbynorm import ClipByNorm

clipper_registry = {
    "by_value":           ClipByValue,
    "by_norm":            ClipByNorm,
}