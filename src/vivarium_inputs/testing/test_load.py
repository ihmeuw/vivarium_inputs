

class DummyLoadComponent:
    def __init__(self, entity_key: str):
        self.entity_key = entity_key

    def setup(self, builder):

        builder.data.load(self.entity_key)
        print(f'data loaded for {self.entity_key}')
