"""This module is a version controlled listing of data that CEAM uses which is not stored in a standard location, like the modelable entity database.
"""

FILES = {
        'Angina Proportions' : {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/angina_props.csv',
            'source': '/snfs1/WORK/04_epi/01_database/02_data/cvd_ihd/04_models/02_misc_data/angina_prop_postMI.csv',
            'owner': 'Catherine O. Johnson <johnsoco@uw.edu>',
        },
        'Age-Specific Fertility Rates': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/ASFR.csv',
            'encoding': 'latin1',
            'source': 'covariate_short_name: Age-Specific Fertility Rate; model_version: 9065',
            'owner': 'HAIDONG WANG <haidong@uw.edu>',
        },
        'Disability Weights': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/dw.csv',
            'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/02_standard/dw.csv',
        },
        'Combined Disability Weights': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/combined_dws.csv',
            'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/03_custom/combined_dws.csv',
        },
        'Systolic Blood Pressure Distributions': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/systolic_blood_pressure/exp_{location_id}_{year_id}_{sex_id}.dta',
            'source': '/share/epi/risk/paf/metab_sbp_interm',
            'encoding': 'latin1',
            'owner': 'Stan Biryukov <stan0625@uw.edu>',
        },
        'Body Mass Index Distributions': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015/bmi/{parameter}/19_{location_id}_{year_id}_{sex_id}.csv',
            'source': '/share/covariates/ubcov/04_model/beta_parameters/8',
            'owner': 'Marissa B. Reitsma <mreitsma@uw.edu>',
        }

}

def auxiliary_file_path(name, **kwargs):
    template_parameters = dict(kwargs)
    template_parameters['j_drive'] = '/home/j'
    raw_path = FILES[name]['path']
    return raw_path.format(**template_parameters)

def open_auxiliary_file(name, **kwargs):
    path = auxiliary_file_path(name, **kwargs)
    encoding = FILES[name].get('encoding')
    return open(path, encoding=encoding)
