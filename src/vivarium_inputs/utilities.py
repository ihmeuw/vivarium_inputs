"""Errors and utility functions for input processing."""
from .globals import gbd


def get_location_id(location_name):
    return {r.location_name: r.location_id for _, r in gbd.get_location_ids().iterrows()}[location_name]


def scrub_gbd_conventions(data, location):
    scrub_location(data, location)
    scrub_sex(data)
    scrub_age(data)
    scrub_year(data)


def scrub_location(data, location):
    if 'location_id' in data.columns:
        data = data.drop('location_id', 'columns')
    data['location'] = location
    return data


def scrub_sex(data):
    if 'sex_id' in data.columns:
        data['sex'] = data['sex_id'].map({1: 'Male', 2: 'Female'})
        data = data.drop('sex_id', 'columns')
    return data


def scrub_age(data):
    if 'age_group_id' in data.columns:
        age_bins = (
            gbd.get_age_bins()[['age_group_id', 'age_group_years_start', 'age_group_years_end']]
                .rename(columns={'age_group_years_start': 'age_group_start',
                                 'age_group_years_end': 'age_group_end'})
        )
        data = data.merge(age_bins, on='age_group_id').drop('age_group_id', 'columns')
    return data


def scrub_year(data):
    if 'year_id' in data.columns:
        data = data.rename(columns={'year_id': 'year_start'})
        data['year_end'] = data['year_start'] + 1
    return data
