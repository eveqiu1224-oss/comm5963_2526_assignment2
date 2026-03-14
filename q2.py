import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier

from utils import load_train_test_datasets
from static import FEATURES, TARGET


def run_decision_tree():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    model = None
    # TODO: Run a classification by constructing a decision tree (Please set the random_state to 5963)
    model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=5963)
    model.fit(train_x, train_y)

    # TODO: Print the train and test accuracy of the model
    train = model.score(train_x, train_y)
    test = model.score(test_x, test_y)
    print(f"Train accuracy: {train:.2%}")
    print(f"Test accuracy: {test:.2%}")

    return model

def show_decision_tree(model_from_part1):
    # TODO: Visualize the decision tree
    plt.figure(figsize=(16, 6), dpi=300)
    plot_tree(
        model_from_part1,
        feature_names=FEATURES,
        class_names=sorted(model.classes_),
        filled=True,
        fontsize=8)
    plt.show()


def run_random_forest():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    # TODO: Run a classification by constructing a random forest (Please set the random_state to 5963)
    rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=3, random_state=5963)
    rf_model.fit(train_x, train_y)

    # TODO: Print the train and test accuracy of the model
    train = rf_model.score(train_x, train_y)
    test = rf_model.score(test_x, test_y)
    print(f"Train accuracy: {train:.2%}")
    print(f"Test accuracy: {test:.2%}")



if __name__ == '__main__':
    print('[Q2][Part 1] Run Decision Tree')
    model = run_decision_tree()
    print('[Q2][Part 2] Visualize the Decision Tree')
    show_decision_tree(model_from_part1=model)
    print('[Q2][Part 3] Run Random Forest')
    run_random_forest()
