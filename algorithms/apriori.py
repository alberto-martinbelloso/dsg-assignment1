import numpy as np
import pandas as pd
from itertools import chain, combinations


def apriori(data_raw, min_support=0.3, min_confidence=0.3):
    data = get_onehot_matrix(data_raw)
    support = []
    for i in range(data.shape[1]):
        support.append(compute_support(data, (i,)))

    result = find_sets_with_support(data, support, min_support)
    result['items'] = result['itemsets'].map(lambda items: list(items))
    metrics = []

    for index, row in result.iterrows():
        if len(row['items']) > 1:
            supports = {'total': row['support']}
            for i in range(len(row['items'])):
                supports[row['items'][i]] = result['support'][row['items'][i]]

            association_rules = create_association_rules(row['itemsets'])

            for rule in association_rules:
                support_xy = supports['total']
                support_x, support_y = find_supports(result, rule)
                confidence = compute_confidence(support_xy, support_x)

                if confidence >= min_confidence:
                    lift = compute_lift(support_xy, support_x, support_y)
                    conviction = compute_conviction(support_y, confidence)
                    metrics.append({'association_rule': rule, 'itemsets': row['items'], 'support': support_xy,
                                    'confidence': confidence, 'lift': lift, 'conviction': conviction})
    metrics = pd.DataFrame(metrics)
    return metrics


def create_association_rules(sets):
    subs = subsets(sets)
    subs.remove(subs[0])
    subs.remove(subs[len(subs)-1])
    ass_rules = []
    itemsets = list(sets)
    for sub in subs:
        subset = list(sub)
        if len(subset) > 0:
            item_list = [e for e in itemsets if e not in subset]
            ass_rules.append([subset, item_list])

    return ass_rules


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def subsets(s):
    return list(map(set, powerset(s)))


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

    return data


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


def find_sets_with_support(data, support, min_support):

    support = np.asarray(support)
    support_dict = {1: support[support >= min_support]}

    # Setup all the atomic items in a dict.
    ary_col_idx = np.arange(data.shape[1])
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}

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


def compute_confidence(support_xy, support_x):
    return support_xy / support_x


def compute_lift(support_xy, support_x, support_y):
    return support_xy / (support_x * support_y)


def compute_conviction(support_x, confidence_xy):
    return (1 - support_x) / (1 - confidence_xy)


def find_supports(result, rule):
    support_x = support_y = -1
    found = 0
    for index1, row1 in result.iterrows():
        if sorted(row1['items']) == sorted(rule[0]):
            support_x = row1['support']
            found += 1
        elif sorted(row1['items']) == sorted(rule[1]):
            support_y = row1['support']
            found += 1
        if found == 2:
            return support_x, support_y
