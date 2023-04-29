"""
https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

"""
import collections
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=42, class_weight='balanced')
iris = load_iris()
print(iris.feature_names)
print(iris.data[:5, :])
print(iris.target_names)
print(collections.Counter(iris.target))

X = np.concatenate([iris.data[20:50, :], iris.data[50+10:100, :], iris.data[100:150, :]])
y = np.concatenate([iris.target[20:50], iris.target[50+10:100], iris.target[100:150]])
clf = clf.fit(X, y)       # tree is in preorder and stored in children_left and children_right
n_class = len(set(y))
class_weights = len(X)/(n_class*np.bincount(y))
print(f'class_weights: {class_weights}')
print(collections.Counter(y))
# out_file = 'a.png'
# with open(out_file, 'w') as f:
#     tree.export_graphviz(clf, out_file=f, feature_names=iris.feature_names, class_names=iris.target_names)
#
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), dpi=200)
tree.plot_tree(clf, ax=axes, max_depth=None, fontsize=5, node_ids=True, filled=True, label='all', proportion=False,
               feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold
impurity = clf.tree_.impurity
value = clf.tree_.value


def retrieve_branches(number_nodes, children_left_list, children_right_list):
    """Retrieve decision tree branches"""

    # Calculate if a node is a leaf
    is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]

    # Store the branches paths
    paths = []

    for i in range(number_nodes):
        if is_leaves_list[i]:
            # Search leaf node in previous paths
            end_node = [path[-1] for path in paths]

            # If it is a leave node yield the path
            if i in end_node:
                output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                yield output

        else:

            # Origin and end nodes
            origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

            # Iterate over previous paths to add nodes
            for index, path in enumerate(paths):
                if origin == path[-1]:
                    paths[index] = path + [end_l]
                    paths.append(path + [end_r])

            # Initialize path in first iteration
            if i == 0:
                paths.append([i, children_left[i]])
                paths.append([i, children_right[i]])

all_branches = list(retrieve_branches(n_nodes, children_left, children_right))
all_branches