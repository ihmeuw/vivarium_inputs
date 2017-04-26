"""This module is a version controlled listing of data that CEAM uses which is not stored in a standard location, like the modelable entity database.
"""

import platform
from os.path import join

from ceam import config

FILES = {
    'Angina Proportions' : {
        'path': 'angina_prop_postMI.csv',
        'source': '/snfs1/WORK/04_epi/01_database/02_data/cvd_ihd/04_models/02_misc_data/angina_prop_postMI.csv',
        'owner': 'Catherine O. Johnson <johnsoco@uw.edu>',
        },
    'Age-Specific Fertility Rates': {
        'path': 'ASFR.csv',
        'encoding': 'latin1',
        'source': 'covariate_short_name: Age-Specific Fertility Rate; model_version: 9065',
        'owner': 'HAIDONG WANG <haidong@uw.edu>',
        },
    'Disability Weights': {
        'path': 'dw.csv',
        'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/02_standard/dw.csv',
        },
    'Combined Disability Weights': {
        'path': 'combined_dws.csv',
        'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/03_custom/combined_dws.csv',
        },
        'Systolic Blood Pressure Distributions': {
            'path': 'systolic_blood_pressure/exp_{location_id}_{year_id}_{sex_id}.dta',
            'source': '/share/epi/risk/paf/metab_sbp_interm',
            'encoding': 'latin1',
            'owner': 'Stan Biryukov <stan0625@uw.edu>',
        },
    'Body Mass Index Distributions': {
        'path': 'bmi/{parameter}/19_{location_id}_{year_id}_{sex_id}.csv',
        'source': '/share/covariates/ubcov/04_model/beta_parameters/8',
        'owner': 'Marissa B. Reitsma <mreitsma@uw.edu>',
        },
    'Fasting Plasma Glucose Distributions': {
        'path': 'fpg/FILE_{location_id}_{year_id}_{sex_id}_OUT.csv',
        'source': '/share/epi/risk/paf/metab_fpg_cont_sll/FILE_[location_id]_[year_id]_[sex_id]_OUT.csv',
        'owner': 'Stan Biryukov <stan0625@uw.edu>',
        },
    'Life Table': {
        'path': 'FINAL_min_pred_ex.csv',
        'source': '/home/j/WORK/10_gbd/01_dalynator/02_inputs/YLLs/usable/FINAL_min_pred_ex.csv',
        },
    'Risk Factor Propensity Correlation Matrices': {
        'path': 'risk_factor_propensity_correlation_matricies/{matrix_variation}',
        'owner': 'Reed Sorenson <rsoren@uw.edu>',
        },
    'Doctor Visit Costs': {
        'path': 'doctor_visit_cost_KEN_20160804.csv',
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },
    'Inpatient Visit Costs': {
        'path': 'inpatient_visit_cost_KEN_20170125.csv',
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },

}

def auxiliary_file_path(name, **kwargs):
    template_parameters = dict(kwargs)
    if platform.system() == 'Windows':
        template_parameters['j_drive'] = 'J:'
    else:
        template_parameters['j_drive'] = '/home/j'
    raw_path = FILES[name]['path']
    auxiliary_data_folder = config.input_data.auxiliary_data_folder
    return join(auxiliary_data_folder, raw_path).format(**template_parameters)

def open_auxiliary_file(name, **kwargs):
    path = auxiliary_file_path(name, **kwargs)
    encoding = FILES[name].get('encoding')
    return open(path, encoding=encoding)
