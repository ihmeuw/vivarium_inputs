"""This module is a version controlled listing of data that CEAM
uses which is not stored in a standard location, like the modelable entity database.
"""

import platform
from os.path import join

AUXILIARY_DATA_FOLDER = "{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/{gbd_round}"

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
        # FIXME: This will go away when we get a consistent access pattern for exposure SDs.
        # 'path': 'systolic_blood_pressure/exp_{location_id}_{sex_id}.dta',
        'path': '/share/costeffectiveness/CEAM/Auxliary_Data/systolic_blood_pressure/exp_{location_id}_{sex_id}.dta',
        'source': '/share/epi/risk/paf/metab_sbp/exposures',
        'encoding': 'latin1',
        'owner': 'Stan Biryukov <stan0625@uw.edu>',
        },
    'Body Mass Index Distributions': {
        'path': 'bmi/{parameter}/19_{location_id}_{year_id}_{sex_id}.csv',
        'source': '/share/covariates/ubcov/04_model/beta_parameters/8',
        'owner': 'Marissa B. Reitsma <mreitsma@uw.edu>',
        },
    'Ensemble Distribution Weights': {
        'path': 'ensemble/weights/{rei_id}.csv',
        'source': '/home/j/WORK/05_risk/ensemble/weights',
        'owner': 'Patrick J Sur <psur2417@uw.edu>, Stan Biryukov <stan0625@uw.edu>',
        },
    'Fasting Plasma Glucose Distributions': {
        'path': 'fpg/FILE_{location_id}_{year_id}_{sex_id}_OUT.csv',
        'source': '/share/epi/risk/paf/metab_fpg_cont_sll/FILE_[location_id]_[year_id]_[sex_id]_OUT.csv',
        'owner': 'Stan Biryukov <stan0625@uw.edu>',
        },
    'Risk Factor Propensity Correlation Matrices': {
        'path': 'risk_factor_propensity_correlation_matricies/location_{location_id}',
        'owner': 'Reed Sorenson <rsoren@uw.edu>',
        },
    'Inpatient Visit Costs': {
        'path': 'ip_cost.csv',
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },
    'Hypertension Drug Costs': {
        'path': 'higashi_drug_costs_20160804.csv',
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },
    'Outpatient Visit Costs': {
        'path': 'op_cost.csv',
        'source': "/snfs1/Project/Cost_Effectiveness/Access_to_care/02_analysis/01_data/op_unit_cost.csv",
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
    'Diarrhea Costs': {
        'path':  'op_ip_diarrhea_cost_all_country_years.csv',
        'source':  '/home/j/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/op_ip_diarrhea_cost_all_country_years.csv',
        'owner': 'Mark Moses <mwm6@uw.edu>',
    },
    'ORS Costs': {
        'path': 'healthcare_access/ors_cost/{location_id}.csv',
        'source':  '/home/j/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/op_ip_diarrhea_cost_all_country_years.csv',
        'owner': 'Mark Moses <mwm6@uw.edu>',
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
