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
          'tables',
      ],
      extras_require={
          'gbd_access': [
              'get_draws',
              'hierarchies',
              'db_tools',
              'db_queries',
              'risk_utils',
          ]
          'testing': [
            'pytest',
            'pytest-mock',
            'hypothesis',
          ]
      }
      )
