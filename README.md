# Data ingestion code for CEAM

## GBD access
This package can provide data ingestion either from precached inputs, in which case there are no special dependencies, or via direct GBD access, in which case it requires several internal IHME packages and access to IHME cluster. To install the extra dependencies create a file caled ~/.pip/pip.conf which looks like this:

    [global]
    extra-index-url = http://dev-tomflem.ihme.washington.edu/simple
    trusted-host = dev-tomflem.ihme.washington.edu

This file tells the pip package management system to check with IHME's internal pypi server for packages. You can then install the optional packages by running this command from inside the ceam-inputs repository:

    pip install .[gbd_access]
