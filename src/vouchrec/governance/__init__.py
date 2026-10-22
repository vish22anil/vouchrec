"""Dynamic Exposure Governor package for post-recommendation safety."""

from vouchrec.governance.config import ExposureGovernorConfig
from vouchrec.governance.exposure_governor import (
    ExposureGovernor,
    GovernanceAuditRecord,
    GovernanceResult,
    ItemTrustEvidence,
    compute_item_trust_evidence,
)

__all__ = [
    "ExposureGovernorConfig",
    "ExposureGovernor",
    "ItemTrustEvidence",
    "GovernanceAuditRecord",
    "GovernanceResult",
    "compute_item_trust_evidence",
]
