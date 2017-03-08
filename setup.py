from setuptools import setup, find_packages


setup(name='ceam_inputs',
        version='0.1',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            'pandas',
            'numpy',
            'scipy',
            'joblib',
            'flufl.lock',
        ],
        extras_require = {
            'gbd_access': [
                'transmogrifier',
                'hierarchies',
                'db_tools',
                ]
        }
     )
