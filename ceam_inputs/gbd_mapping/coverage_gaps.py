from typing import Tuple, Union

from .templates import meid, rid, Cause, Restrictions, Levels, Tmred, ExposureParameters, GbdRecord
from .causes import causes

class CoverageGap(GbdRecord):
    """Container for coverage gap GBD ids and metadata."""
    __slots__ = ('name', 'gbd_id', 'distribution', 'affected_causes', 'restrictions', 'levels',
                 'exposure', 'relative_risk', 'population_attributable_fraction', )

    def __init__(self,
                 name: str,
                 gbd_id: Union[meid, rid, None],
                 affected_causes: Tuple[Cause, ...],
                 restrictions: Restrictions,
                 distribution: str = 'dichotomous',
                 levels: Levels = None,
                 exposure: str = None,
                 relative_risk: str = None,
                 population_attributable_fraction: str=None, ):
        super().__init__()
        self.name = name
        self.gbd_id = gbd_id
        self.distribution = distribution
        self.affected_causes = affected_causes
        self.restrictions = restrictions
        self.levels = levels
        self.exposure = exposure
        self.relative_risk = relative_risk
        self.population_attributable_fraction = population_attributable_fraction


class CoverageGaps(GbdRecord):
    """Container for coverage gaps."""
    __slots__ = ('lack_of_exposure_to_antiretroviral_therapy', 'low_measles_vaccine_coverage_first_dose')

    def __init__(self,
                 lack_of_exposure_to_antiretroviral_therapy: CoverageGap,
                 low_measles_vaccine_coverage_first_dose: CoverageGap, ):
        super().__init__()
        self.lack_of_exposure_to_antiretroviral_therapy = lack_of_exposure_to_antiretroviral_therapy
        self.low_measles_vaccine_coverage_first_dose = low_measles_vaccine_coverage_first_dose


coverage_gaps = CoverageGaps(
    lack_of_exposure_to_antiretroviral_therapy=CoverageGap(
        name='lack_of_exposure_to_antiretroviral_therapy',
        gbd_id=None,
        distribution='categorical',
        affected_causes=(causes.hiv_aids_resulting_in_other_diseases, ),
        restrictions=Restrictions(
            male_only=False,
            female_only=False,
            yll_only=False,
            yld_only=False,
        ),
        levels=Levels(
            cat1='exposed',
            cat2='unexposed',
        ),
        exposure='HIV Positive Antiretroviral Therapy Exposure',
        relative_risk='HIV Positive Antiretroviral Therapy Relative Risk',
        population_attributable_fraction='HIV Positive Antiretroviral Therapy PAF'
    ),
    low_measles_vaccine_coverage_first_dose=CoverageGap(
        name='low_measles_vaccine_coverage_first_dose',
        gbd_id=rid(318),
        distribution='categorical',
        affected_causes=(causes.measles, ),
        restrictions=Restrictions(
            male_only=False,
            female_only=False,
            yll_only=False,
            yld_only=False,
        ),
        levels=Levels(
            cat1='exposed',
            cat2='unexposed',
        ),
    )
)

