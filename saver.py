from collections import defaultdict


def split_testset(test_sets, n=50):
    dial_ids = []
    split_dict = defaultdict(list)

    splits = []

    for item in test_sets:
        dial_ids.append(item['ID'])

    dial_ids = list(set(dial_ids))

    for idx, dial_id in enumerate(dial_ids):
        split_dict[idx % n].append(dial_id)

    for i in range(n):
        splits.append([test_sets[j] for j in range(len(test_sets))
                      if test_sets[j]['ID'] in split_dict[i]])

    return splits
