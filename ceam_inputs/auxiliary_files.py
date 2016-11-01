"""This module is a version controlled listing of data that CEAM uses which is not stored in a standard location, like the modelable entity database.
"""

FILES = {
        'Angina Proportions' : {
            'path': '{j_drive}/Project/Cost_Effectiveness/dev/data_processed/angina_props.csv',
            'source': '/snfs1/WORK/04_epi/01_database/02_data/cvd_ihd/04_models/02_misc_data/angina_prop_postMI.csv',
            'owner': 'Catherine O. Johnson <johnsoco@uw.edu>',
        },
        'Age-Specific Fertility Rates': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/ASFR.csv',
            'encoding': 'latin1',
            'source': 'covariate_short_name: Age-Specific Fertility Rate; model_version: 9065',
            'owner': 'HAIDONG WANG <haidong@uw.edu>',
        },
        'Disability Weights': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/dw.csv',
            'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/02_standard/dw.csv',
        },
        'Combined Disability Weights': {
            'path': '{j_drive}/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/combined_dw.csv',
            'source': '/home/j/WORK/04_epi/03_outputs/01_code/02_dw/02_standard/combined_dw.csv',
        },

}

def open_auxiliary_file(name):
    raw_path = FILES[name]['path']
    path = raw_path.format(j_drive='/home/j')
    encoding = FILES[name].get('encoding')
    return open(path, encoding=encoding)
