import pandas as pd
import numpy as np
from math import log2


def calculate_entropy(y):
    classes = np.unique(y)
    entropy = 0
    for cls in classes:
        p_cls = np.sum(y == cls) / len(y)
        entropy -= p_cls * log2(p_cls)
    return entropy


def calculate_information_gain(data, attribute, target):
    original_entropy = calculate_entropy(data[target])
    values = np.unique(data[attribute])
    weighted_sum = 0
    for value in values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        subset_entropy = calculate_entropy(subset[target])
        weighted_sum += weight * subset_entropy
    information_gain = original_entropy - weighted_sum
    return information_gain

def decision_tree_learning(data, attributes, target):
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]
    elif len(attributes) == 0:
        return np.argmax(np.unique(data[target]))
    else:
        gains = [calculate_information_gain(data, attribute, target) for attribute in attributes]
        best_attr_index = np.argmax(gains)
        best_attr = attributes[best_attr_index]
        tree = {best_attr: {}}
        attributes = [attr for attr in attributes if attr != best_attr]
        for value in np.unique(data[best_attr]):
            subset = data[data[best_attr] == value]
            tree[best_attr][value] = decision_tree_learning(subset, attributes, target)
        return tree

data = pd.DataFrame({
    'a1': [1, 1, 0, 1, 1],
    'a2': [0, 0, 1, 1, 1],
    'a3': [0, 1, 0, 1, 0],
    'dec': [0, 0, 0, 1, 1]
})

attributes = ['a1', 'a2', 'a3']
target = 'dec'

tree = decision_tree_learning(data, attributes, target)
print(tree)
