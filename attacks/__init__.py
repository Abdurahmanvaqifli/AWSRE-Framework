"""
AWSRE Attack Simulation Package
"""

from attacks.pipeline import (
    AttackPipeline,
    AttackResult,
    AttackStep,
    SUPPORTED_ATTACKS,
    apply_attack,
    apply_combined_attacks,
    normalize_attack_name,
)

__all__ = [
    "AttackPipeline",
    "AttackResult",
    "AttackStep",
    "SUPPORTED_ATTACKS",
    "apply_attack",
    "apply_combined_attacks",
    "normalize_attack_name",
]
