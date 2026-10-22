"""Controlled review manipulation attack simulation and evaluation."""

from vouchrec.attacks.config import AttackConfig, AttackObjective, AttackStrategy, TargetSelection
from vouchrec.attacks.evaluation import (
    AttackEvaluationReport,
    AttackExperimentRunner,
    evaluate_target_item_impact,
    evaluate_trust_detection,
)
from vouchrec.attacks.simulator import AttackSimulationResult, AttackSimulator
from vouchrec.attacks.strategies import (
    AverageAttack,
    BandwagonAttack,
    BaseAttackStrategy,
    RandomAttack,
    create_attack_strategy,
)

__all__ = [
    "AttackConfig",
    "AttackStrategy",
    "AttackObjective",
    "TargetSelection",
    "BaseAttackStrategy",
    "RandomAttack",
    "AverageAttack",
    "BandwagonAttack",
    "create_attack_strategy",
    "AttackSimulator",
    "AttackSimulationResult",
    "AttackEvaluationReport",
    "AttackExperimentRunner",
    "evaluate_target_item_impact",
    "evaluate_trust_detection",
]
