from fedlearning.compressors.signsgd import SignSGDCompressor, OptimalStoSignSGDCompressor
from fedlearning.compressors.ternary import TernaryCompressor
from fedlearning.compressors.randomsparse import RandomSparsifier
from fedlearning.compressors.attacker_ternary import AttackerTernaryCompressor
from fedlearning.compressors.attacker_randomsparse import AttackerRandomSparsifier

compressor_registry = {
    "signSGD":          SignSGDCompressor,
    "stosignsgd":       OptimalStoSignSGDCompressor,
    "randomsparse":     RandomSparsifier,
    "ternary":          TernaryCompressor,
    "attacker_ternary": AttackerTernaryCompressor,
    "attacker_randomsparse": AttackerRandomSparsifier
}
