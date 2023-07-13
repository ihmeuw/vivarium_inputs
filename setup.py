#!/usr/bin/env python
import os

from setuptools import find_packages, setup

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_inputs", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        "numpy",
        "scipy",
        "pandas",
        "click",
        "joblib",
        "tables",
        "vivarium>=1.2.1",
        "gbd_mapping>=3.1.0, <4.0.0",
        "loguru",
    ]

    setup_requires = ["setuptools_scm"]

    data_requires = ["vivarium-gbd-access>=3.0.7, <4.0.0", "core-maths"]

    test_requirements = [
        "pytest",
        "pytest-mock",
        "hypothesis",
    ]

    doc_requirements = [
        "sphinx>=4.0",
        "sphinx-rtd-theme",
    ]

    setup(
        name=about["__title__"],
        description=about["__summary__"],
        long_description=long_description,
        url=about["__uri__"],
        author=about["__author__"],
        author_email=about["__email__"],
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            "docs": doc_requirements,
            "test": test_requirements,
            "data": data_requires,
            "dev": doc_requirements + test_requirements + data_requires,
        },
        zip_safe=False,
        use_scm_version={
            "write_to": "src/vivarium_inputs/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
    )
