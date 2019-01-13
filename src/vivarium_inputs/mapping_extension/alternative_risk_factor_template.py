from gbd_mapping.base_template import GbdRecord
from gbd_mapping.risk_factor_template import RiskFactor


class AlternativeRiskFactor(RiskFactor):
    """Container for alternative GBD risk factor definitions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AlternativeRiskFactors(GbdRecord):
    """Holder of alternative risk factors."""
    __slots__ = ('child_wasting', 'child_underweight', 'child_stunting', )

    def __init__(self,
                 child_stunting: AlternativeRiskFactor,
                 child_underweight: AlternativeRiskFactor,
                 child_wasting: AlternativeRiskFactor,):
        self.child_stunting = child_stunting
        self.child_underweight = child_underweight
        self.child_wasting = child_wasting
