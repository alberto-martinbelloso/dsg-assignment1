from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def apriori(data_raw):
    # pprint(data_raw)
    data, features_names = get_onehot_matrix(data_raw)

    min_support = 0.3
    min_confidence = 0.3

    support = []
    for i in range(data.shape[1]):
        support.append(compute_support(data, (i,)))

    # plot_data(features_names, support)
    res_df = find_right_sets(data, support, min_support)
    for i in range(37, len(res_df)):
        print([features_names[i] for i in res_df.itemsets[i]], '\t has support {:.4f}'.format(res_df.support[i]))

    return res_df


def get_onehot_matrix(data_raw):
    num_obs = len(data_raw)
    bad_feature_names = ['', 'all- purpose']

    # Make a list of all the items
    all_items = []
    for i in range(num_obs):
        all_items += data_raw[i]['items']

    # Find the unique item names
    features_names = list(np.unique(all_items))

    # Remove bad item names
    for bad_feature in bad_feature_names:
        features_names.remove(bad_feature)

    # Create a one-hot matrix
    num_features = len(features_names)
    data = np.zeros([num_obs, num_features])
    for i in range(num_obs):
        for item in data_raw[i]['items']:
            if item not in bad_feature_names:
                j = features_names.index(item)
                data[i,j] = 1

    return data, features_names


def compute_support(X, combi):
    rule = X[:, combi].all(axis=1)
    support = rule.sum() / X.shape[0]
    return support


def candidate_generation(old_combinations):

    items_types_in_previous_step = np.unique(old_combinations.flatten())

    for old_combination in old_combinations:
        max_combination = max(old_combination)
        for item in items_types_in_previous_step:
            if item > max_combination:
                res = tuple(old_combination) + (item,)
                yield res


def find_right_sets(data, support, min_support):

    support = np.asarray(support)
    support_dict = {1: support[support >= min_support]}

    # Setup all the atomic items in a dict.
    ary_col_idx = np.arange(data.shape[1])
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    rows_count = float(data.shape[0])

    itemset_len = 1
    while itemset_len:
        next_itemset_len = itemset_len + 1
        combin = candidate_generation(itemset_dict[itemset_len])
        frequent_items = []
        frequent_items_support = []

        for c in combin:
            support = compute_support(data, c)

            if support >= min_support:
                frequent_items.append(c)
                frequent_items_support.append(support)

        if frequent_items:
            itemset_dict[next_itemset_len] = np.array(frequent_items)
            support_dict[next_itemset_len] = np.array(frequent_items_support)
            itemset_len = next_itemset_len
        else:
            itemset_len = 0

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    res_df = res_df.reset_index(drop=True)

    return res_df


def compute_confidence():
    """ The confidence for `X->Y` is the likelihood that `Y` is purchased, if `X` is purchased.
        This is the same as the conditional probability.
        $conf(X \rightarrow Y) = \frac{supp(X \cup Y)}{supp(X)}$
    """
    pass


def compute_lift():
    """
        $lift(X\rightarrow Y) = \frac{supp(X \cup Y)}{supp(X) supp(Y)}$
    """
    pass


def compute_conviction():
    """ How much better than chance is this association?

        $conv(X \rightarrow Y) = \frac{1-supp(Y)}{1-conf(X->Y)}$
    """
    pass


def plot_data(features_names, support):
    plt.figure(figsize=(10,4))
    g = sns.barplot(x=features_names, y=support, color='b')
    plt.xticks(rotation=90)
    plt.show()
