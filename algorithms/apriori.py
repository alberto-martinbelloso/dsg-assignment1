import numpy as np
import pandas as pd
from itertools import chain, combinations


def apriori(data_raw, min_support=0.3, min_confidence=0.3):
    data, features_names = get_onehot_matrix(data_raw)

    frequent_itemsets = find_frequent_itemsets(data, min_support)

    result = []
    for index, row in frequent_itemsets.iterrows():
        if len(row['items']) > 1:
            association_rules = create_association_rules(row['itemsets'])
            metrics = generate_metrics_df(association_rules, features_names, row['support'], frequent_itemsets, min_confidence)
            for metric in metrics:
                result.append(metric)

    return pd.DataFrame(result)


def create_association_rules(sets):
    subs = subsets(sets)
    subs.remove(subs[0])
    subs.remove(subs[len(subs) - 1])
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
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


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
                data[i, j] = 1
    features_names[8] = 'detergent'

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


# already uses apriori pruning principle
def find_frequent_itemsets(data, min_support):
    support = []
    for i in range(data.shape[1]):
        support.append(compute_support(data, (i,)))
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

    res_df['items'] = res_df['itemsets'].map(lambda items: list(items))
    return res_df


def compute_confidence(support_xy, support_x):
    return support_xy / support_x


# Pattern evaluation - correlation
def compute_lift(confidence, support_y):
    return confidence / support_y


def compute_conviction(support_x, confidence_xy):
    return (1 - support_x) / (1 - confidence_xy)


def compute_kulczynski(confidence_xy, confidence_yx):
    return 0.5 * (confidence_xy + confidence_yx)


def compute_ir(support_a, support_b, support_ab):
    return abs(support_a - support_b) / (support_a + support_b - support_ab)


def find_supports(frequent_itemsets, rule):
    support_x = support_y = -1
    found = 0
    for index1, row1 in frequent_itemsets.iterrows():
        if sorted(row1['items']) == sorted(rule[0]):
            support_x = row1['support']
            found += 1
        elif sorted(row1['items']) == sorted(rule[1]):
            support_y = row1['support']
            found += 1
        if found == 2:
            return support_x, support_y


def build_string(rule, features_names):
    r = '{'
    first = rule[0]
    multiple = False
    if len(first) > 1:
        multiple = True

    count = 0
    for el in first:
        r += features_names[el]
        if multiple and count != len(first)-1:
            r += ', '
        count += 1

    r += '} -> {'
    second = rule[1]
    multiple = False
    if len(second) > 1:
        multiple = True

    count = 0
    for el in second:
        r += features_names[el]
        if multiple and count != len(second)-1:
            r += ', '
        count += 1
    r += '}'

    return r


def generate_metrics_df(association_rules, features_names, support_xy, frequent_itemsets, min_confidence):
    metrics = []
    for rule in association_rules:
        r = build_string(rule, features_names)
        support_x, support_y = find_supports(frequent_itemsets, rule)
        confidence_xy = compute_confidence(support_xy, support_x)
        confidence_yx = compute_confidence(support_xy, support_y)

        if confidence_xy >= min_confidence:
            lift = compute_lift(confidence_xy, support_y)
            conviction = compute_conviction(support_y, confidence_xy)
            kulczynski = compute_kulczynski(confidence_yx, confidence_xy)
            ir = compute_ir(support_x, support_y, support_xy)
            metrics.append({'rule': rule,'a_rule':r, 'support': round(support_xy, 2),
                            'support_x': round(support_x, 2), 'support_y': round(support_y,2), 'kulczynski': round(kulczynski, 2),
                            'confidence': round(confidence_xy, 2), 'lift': round(lift, 2),
                            'conviction': round(conviction, 2), 'ir': round(ir, 2)})
    return metrics
