from .mean import Mean
from .mean_attack import Mean_Attack
from .vote import Vote, PluralityVote
from .centeredclipping import CenteredClipping
from .median import Median
from .signguard import SignGuard
from .trimmedmean import TrimmedMean
from .krum import Multikrum

aggregator_registry = {
    "mean":             Mean,
    "vote":             Vote,
    "plurality_vote":   PluralityVote,
    "centered_clipping": CenteredClipping,
    "median":           Median,
    "sign_guard":       SignGuard,
    "trimmed_mean":     TrimmedMean,
    "mean_attack":     Mean_Attack,
    "krum":             Multikrum
   
}