import pandas as pd
import numpy as np
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
np.random.seed(42)

def train_n_test(X, y, max_depth=17):
    s = int(0.7 * len(X))  # split
    mymodel = DecisionTree(criterion="information_gain", max_depth=max_depth)

    # Training
    X_train = pd.DataFrame(X[0:s]).reset_index(drop=True)
    y_train = pd.Series(y[0:s], dtype="category").reset_index(drop=True)
    mymodel.fit(X_train, y_train)

    # Test
    X_test = pd.DataFrame(X[s:]).reset_index(drop=True)
    y_test = pd.Series(y[s:]).reset_index(drop=True)

    y_pred = mymodel.predict(X_test)

    print("Accuracy:", accuracy(pd.Series(y_pred), y_test))

    for cls in y_test.unique():
        print(f'Precision -- class {cls} =', precision(y_pred, y_test, cls))
        print(f'Recall -- class {cls} =:', recall(y_pred, y_test, cls))


# main ---------
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

print("\nQuestion 2) a)\n")
train_n_test(X, y)
print("\nEnd of Question 2) a)\n")
