def feedback(vector, weights, direction):
    new_vector = []
    for idx, x in enumerate(vector):
        new_vector.append(x + (weights[idx] * direction))
    return new_vector