import os
import tarfile
import json

import pandas as pd

from ceam_inputs.data_artifact import split_entity_path

class ArtifactReader:
    def __init__(self, path):
        self.path = path

    def data_container(self, entity_path):
        with tarfile.open(self.path, 'r') as f:
            root = f.getnames()[0]
            entity_type, entity_name = split_entity_path(entity_path)
            path = os.path.join(root, entity_type)
            if entity_name:
                path = os.path.join(path, entity_name)
            measure_names = [n for n in f.getnames() if n.startswith(path) and n != path]
            measures = {}
            for name in measure_names:
                if name.endswith(".json"):
                    data = json.load(f.extractfile(name))
                elif name.endswith(".hdf"):
                    data = pd.read_hdf(pd.HDFStore(
                           "data.h5",
                           mode="r",
                           driver="H5FD_CORE",
                           driver_core_backing_store=0,
                           driver_core_image=f.extractfile(name).read()
                           ))

                else:
                    raise ValueError("File must be json or hdf")
                measures[os.path.basename(os.path.splitext(name)[0])] = data
            import pdb; pdb.set_trace()
            return measures

ArtifactReader('/tmp/test_artifact.tgz').data_container('risk_factor.unsafe_water_source')
