def feedback(vector, weights, direction):
    new_vector = []
    for idx, x in enumerate(vector):
        new_vector.append(x + (weights[idx] * .1 * direction))
    return new_vector