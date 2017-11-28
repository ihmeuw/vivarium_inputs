"""This module is a version controlled listing of data that CEAM
uses which is not stored in a standard location, like the modelable entity database.
"""

import platform
from os.path import join

AUXILIARY_DATA_FOLDER = "{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/{gbd_round}"

FILES = {
    'Disability Weights': {
        'path': 'dw.csv',
        'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/02_standard/dw.csv',
        },
    'Combined Disability Weights': {
        'path': 'combined_dws.csv',
        'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/03_custom/combined_dws.csv',
        },
    'Ensemble Distribution Weights': {
        'path': 'ensemble/weights/{rei_id}.csv',
        'source': '/home/j/WORK/05_risk/ensemble/weights',
        'owner': 'Patrick J Sur <psur2417@uw.edu>, Stan Biryukov <stan0625@uw.edu>',
        },
    'Risk Factor Propensity Correlation Matrices': {
        'path': 'risk_factor_propensity_correlation_matricies/location_{location_id}',
        'owner': 'Reed Sorenson <rsoren@uw.edu>',
        },
    'Inpatient Visit Costs': {
        'path': 'costs/healthcare_entities/inpatient.hdf',
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },
    'Outpatient Visit Costs': {
        'path': 'costs/healthcare_entities/outpatient.hdf',
        'source': "/snfs1/Project/Cost_Effectiveness/Access_to_care/02_analysis/01_data/op_unit_cost.csv",
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },
    'Hypertension Drug Costs': {
        'path': 'costs/treatment_technologies/hypertension_drugs.hdf',
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },
    'Ors Exposure': {
        'path': 'diarrhea_ors/exposure/{location_id}.csv',
        'source': '/share/epi/risk/bmgf/exp/diarrhea_ors/{location_id}.csv',
        'owner': 'Kelly Cercy <kcercy@uw.edu>; Dietary Risk Factors Team',
    },
    'Ors Relative Risks': {
        'path': 'diarrhea_ors/diarrhea_ors_rrs.csv',
        'source': '/share/epi/risk/bmgf/rr/diarrhea_ors/1.csv',
        'owner': 'Kelly Cercy <kcercy@uw.edu>; Dietary Risk Factors Team',
    },
    'Ors Pafs': {
        'path': 'diarrhea_ors/pafs/paf_yll_{location_id}.csv',
        'source': '/share/epi/risk/bmgf/paf/diarrhea_ors/paf_yll_{location_id}.csv',
        'owner': 'Kelly Cercy <kcercy@uw.edu>; Dietary Risk Factors Team',
    },
    'Mediation Factors': {
        'path': 'mediation_matrix_corrected.csv',
        'source': '/home/j/WORK/05_risk/mediation/mediation_matrix_corrected.csv',
        'owner': 'Kelly Cercy <kcercy@uw.edu>; Dietary Risk Factors Team',
    },
    'Rota Vaccine Protection': {
        'path': 'rota_protection_draws.csv',
        # FIXME: Everett to clean up source code after distribution is chosen and put code in CEAM
        'source': '',
        'owner': 'Everett Mumford <emumford@uw.edu>',
    },
    'Rota Vaccine RRs': {
        'path': 'rota_vaccine_rrs.csv',
        # FIXME: Everett to clean up source code after distribution is chosen and put code in CEAM
        'source': '',
        'owner': 'Everett Mumford <emumford@uw.edu>',
    },
    'Risk Data': {
        'path': 'risk_variables.xlsx',
        'source': '/home/j/WORK/05_risk/central/documentation/GBD\ 2016/risk_variables.xlsx',
        'owner': 'Kelly Cercy <kcercy@uw.edu>'
    }
}


def auxiliary_file_path(name, **kwargs):
    template_parameters = dict(kwargs)
    if platform.system() == 'Windows':
        template_parameters['j_drive'] = 'J:'
    else:
        template_parameters['j_drive'] = '/home/j'
    raw_path = FILES[name]['path']
    return join(AUXILIARY_DATA_FOLDER, raw_path).format(**template_parameters), FILES[name].get('encoding')


def open_auxiliary_file(name, **kwargs):
    path = auxiliary_file_path(name, **kwargs)
    encoding = FILES[name].get('encoding')
    return open(path, encoding=encoding)
