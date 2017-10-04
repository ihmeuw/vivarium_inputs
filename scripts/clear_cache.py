from joblib.memory import Memory

from ceam_inputs.util import get_cache_directory, get_input_config

if __name__ == '__main__':
    memory = Memory(cachedir=get_cache_directory(get_input_config()), verbose=1)
    memory.clear()
