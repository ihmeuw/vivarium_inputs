from joblib.memory import Memory

from ceam_inputs.util import get_cache_directory

if __name__ == '__main__':
    memory = Memory(cachedir=get_cache_directory(), verbose=1)
    memory.clear()
