class VectorGenerator:
    def generate_vectors(self, outcomes, n_generate):
        return [outcome.get_vector() for outcome in outcomes]
