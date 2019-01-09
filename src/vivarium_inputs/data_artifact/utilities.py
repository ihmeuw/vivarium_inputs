import pkg_resources


def get_versions():
    libraries = ['vivarium', 'vivarium_inputs', 'vivarium_public_health', 'gbd_mapping']
    return {k: pkg_resources.get_distribution(k).version for k in libraries}
