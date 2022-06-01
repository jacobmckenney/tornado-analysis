"""
Jacob McKenney & Luke Sala
"""

TREE_TUNING_QUICK = {
    'splitter': ['best'],
    'max_depth': [1, 3],
    'min_samples_leaf': [9, 10],
    'min_weight_fraction_leaf': [0.1, 0.2],
    'max_features': ['auto'],
    'max_leaf_nodes': [10, 20]
}

TREE_TUNING_FULL = {
    'splitter': ['best', 'random'],
    'max_depth': [1, 3, 5, 7, 9, 11, 12],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_weight_fraction_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
    'max_features': ['auto', 'log2', 'sqrt'],
    'max_leaf_nodes': [10, 20, 30, 40, 50, 60, 70, 80, 90]
}

KNEIGHBORS_TUNING_QUICK = {
    'n_neighbors': [2, 3, 4],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

KNEIGHBORS_TUNING_FULL = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 5]
}
