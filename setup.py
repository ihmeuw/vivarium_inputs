from setuptools import setup, find_packages


setup(name='ceam_inputs',
        version='0.1',
        packages=find_packages(),
        include_package_data=True,
        package_index='http://dev-tomflem.ihme.washington.edu/simple/',
        install_requires=[
            'pandas',
            'numpy',
            'scipy',
            'joblib',
            'flufl.lock',
            'transmogrifier',
            'hierarchies',
            'db_tools',
        ]
     )
