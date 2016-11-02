#!/usr/bin/env python
# ~/ceam/setup.py

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
            'db_tools',
        ],
        dependency_links=[
            'git+ssh://git@stash.ihme.washington.edu:7999/cp/db_tools.git#egg=db_tools-999'',
        ]
     )


# End.
