"""
Implements helper functions for executing spatial reference game environments.
"""

def above_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["y"] < candidate["attributes"]["y"]
def right_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["x"] < candidate["attributes"]["x"]
def left_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["x"] > candidate["attributes"]["x"]
def below_fn(sources, candidate):
    if len(sources) != 1:
        return False
    return sources[0]["attributes"]["y"] > candidate["attributes"]["y"]

FUNCTIONS = {
    "spatial_simple": [
        ("above", above_fn)
    ],
    "spatial_complex": [
        ("above", above_fn),
        ("right", right_fn),
        ("left", left_fn),
        ("below", below_fn)
    ]
}

