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
        "numpy<2.0.0",
        "scipy",
        "pandas",
        "click",
        "joblib",
        "tables",
        "vivarium>=1.2.1",
        "gbd_mapping>=4.0.0, <5.0.0",
        "loguru",
    ]

    setup_requires = ["setuptools_scm"]

    data_requires = ["vivarium-gbd-access>=4.1.0, <5.0.0", "core-maths"]

    lint_requirements = ["black==22.3.0", "isort"]

    test_requirements = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "hypothesis",
    ]

    doc_requirements = [
        "sphinx>=7.0, <8.0",
        "sphinx-rtd-theme",
        "sphinx-autodoc-typehints",
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
            "dev": doc_requirements + test_requirements + data_requires + lint_requirements,
        },
        zip_safe=False,
        use_scm_version={
            "write_to": "src/vivarium_inputs/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
    )
