#!/usr/bin/env python
import os

from setuptools import setup, find_packages

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_inputs", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'numpy',
        'scipy',
        'pandas',
        'click',
        'joblib',
        'tables',
        'vivarium',
        'vivarium_public_health',
        'gbd_mapping',
    ]

    data_requires = [
        'vivarium-gbd-access>=1.0.4',
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
        'hypothesis',
    ]

    doc_requirements = [
        'sphinx',
        'sphinx-autodoc-typehints',
        'sphinx-rtd-theme',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            'docs': doc_requirements,
            'test': test_requirements,
            'data': data_requires,
            'dev': doc_requirements + test_requirements + data_requires,
        },

        entry_points='''
            [console_scripts]
            build_artifact=vivarium_inputs.data_artifact.cli:build_artifact
        ''',

        zip_safe=False,
    )
