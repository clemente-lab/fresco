class VectorGenerator:
    def generate_vectors(self, outcomes):
        return [outcome.get_vector() for outcome in outcomes]
