import os
import sys, getopt
import numpy as np
from sklearn import datasets
from sklearn import model_selection
import pandas as pd
import time

verbose = False
feature_names = []
target_names = []
header = []
method = "gini"


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
        self.value_is_numerical = is_numeric(self.value)

    def match(self, example):
        """ Takes a single example (a single row) and compares 
            it with the feature value in this question.
            
            Returns: Bool
        """
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # Helper method to print in a readable format.
        condition = "=="
        if self.value_is_numerical:
            condition = ">="
        return "Is %s %s %s?" % (feature_names[self.column], condition, str(self.value))


class Leaf_Node:
    """
    A Leaf node classifies data.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """
    A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def entropy(rows):

    counts = class_counts(rows)
    impurity = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl * np.log2(prob_of_lbl)
    return impurity


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty, method = gini):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    
    if method == "gini":
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
    elif method == "entropy":
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    global method
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    if method == "gini": current_uncertainty = gini(rows) 
    else: current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty, method)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """
    if verbose: print("Building tree...")
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf_Node(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def classify(row, node, verbose=False):
    """ Recursively search
    """
    if isinstance(node, Leaf_Node):
        return node.predictions
    
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function. <-- Haha - yea, right."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf_Node):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "|.")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "|.")


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def evaluate(my_tree, data_test):
    acc = 0
    for row in data_test:
        if acc >= len(target_names):
            pred = classify(row, my_tree, False)
        else:
            pred = classify(row, my_tree, True)
            if verbose: print ("Actual: %s. Predicted: %s" % (row[-1], print_leaf(pred)))

        if row[-1] in pred:
            acc += 1
        else:
            if verbose: print("Wrong prediction. Actual: %s. Predicted: %s" % (row[-1], print_leaf(pred)))
    return acc/data_test.shape[0]


def prepare_data(loadFromFile, inputFile):
    global header
    global feature_names
    global target_names
    global verbose

    if loadFromFile:
        df = pd.read_csv(inputFile) 
        data = df.get_values()
        header = [
            "State", "Account Length", "Area Code", "International Plan", "Voice mail plan", 
            "Total day minutes", "Total day calls", "Total day charge", 
            "Total eve minutes", "Total eve calls", "Total eve charge",
            "Total night minutes", "Total night calls", "Total night charge",
            "Total intl minutes", "Total intl calls", "total intl charge",
            "Customer service calls", "Churn"
            ]
        feature_names = header
        target_names = ["False", "True"]

    else:
        iris = datasets.load_iris()
        data = np.concatenate([iris.data, iris.target[:,None]], axis=-1) 
        iris.data[:, :2]
        # Column labels.
        # These are used only to print the tree.
        header = ["color", "diameter", "label"]
        feature_names = iris.feature_names
        target_names = iris.target_names
    return data


def partition_data(data):
    step_len = 10
    data_test = data[::step_len,:]
    np.random.shuffle(data_test)
    data_train = np.delete(data, [step_len*i for i in range(data_test.shape[0])], axis=0)#.reshape([-1, data.shape[1]])

    if verbose:
        print('Feature Names:', feature_names)
        print('Target Names: ', target_names)
        print('data\t\t', data.shape)
        print('data_test\t', data_test.shape)
        print('data_train\t', data_train.shape)
        print('data_test shape[0]\t', data_test.shape[0])
        print('Example data')
        print(data_train[::15,:])

    return data_train, data_test


def main(argv):
    start = time.time()
    validation = False
    loadFromFile = False
    inputFile = ''
    global verbose
    global method

    try:
        opts, _ = getopt.getopt(argv, "hHvc:m:i:", ['help', 'verbose', 'method', 'input', 'validation'])
    except getopt.GetoptError:
        print("decision-tree.py\n"
            "Please use -h or --help for more information")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print("decision-tree.py\n"
                    "-v or --verbose for turning verbose on.\n"
                    "-m or --method <gini> or <entropy> for choosing either gini impurity or entropy for calculating information gain, i.e. -m gini or -m entropy\n"
                    "-i or --input for specifying the telecom_churn input file, i.e. -i <file_path>")
            sys.exit()
        elif opt in ('-v', '--verbose'):
            verbose = True
        elif opt in ('-m', '--method'):
            if arg == "gini" or "entropy":
                method = arg
            else:
                print("Unkown method. Please use either <gini> or <entropy> for option -m")
                sys.exit(2)
        elif opt in ('-i', '--input'):
            loadFromFile = True
            if arg == '':
                inputFile = "C:/Users/alexk/OneDrive/Dokumenter/Programming/dsg-assignment1/data/telecom_churn.csv"
            else:
                inputFile = arg
            assert os.path.isfile(inputFile), 'Data not found. Make sure you wrote the correct file path!'
        elif opt in ('-c' or '--validation'):
            validation = True
            try:
                n_split = int(arg)
            except Exception:
                print("Please type a number when using option -c or --validation")
                sys.exit(2)

    data = prepare_data(loadFromFile, inputFile)
    # k-Fold cross-validation.
    if validation:
        kf = model_selection.KFold(n_split, True)
        acc = []
        
        for data_train, data_test in kf.split(data):
            my_tree = build_tree(data[data_train])
            if verbose: print_tree(my_tree)
            acc.append(evaluate(my_tree, data[data_test]))
            print("Accuracy: %0.2f" % acc[-1])
        print("Average accuracy: %0.2f (+/- %0.2f)" % (np.mean(acc), np.std(acc)))
    else:
        data_train, data_test = partition_data(data)
        my_tree = build_tree(data_train)
        if verbose: print_tree(my_tree)
        print("Accuracy: %s" % (evaluate(my_tree, data_test)))

    print("Finished execution in: %s" % (time.time() - start))


if __name__ == '__main__':
    main(sys.argv[1:])

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting
# - add support for regression
