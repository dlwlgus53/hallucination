import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from utils.helper import PreviousStateRecorder
from utils.typo_fix import typo_fix
from config import CONFIG

from utils.sql import sql_pred_parse, sv_dict_to_string
from evaluate_metrics import evaluate
import pdb

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--run_log', type=str, required=True,
                    help="running log filename")
parser.add_argument('--test_fn', type=str, default="./data/mw21_100p_test.json",
                    help="running log filename")
parser.add_argument('--mwz_ver', type=str, default="2.1",
                    choices=['2.1', '2.4'], help="version of MultiWOZ")
args = parser.parse_args()


# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
else:
    ontology_path = CONFIG["ontology_24"]

with open(ontology_path) as f:
    ontology = json.load(f)


def eval(running_log, test_set, turn=-1, use_gold=False):
    result_dict = defaultdict(list)  # use to record the accuracy
    prediction_recorder = PreviousStateRecorder()  # state recorder
    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0
    if len(running_log) != len(test_set):
        print("running log and test set are not aligned")
        print("this should be working as debug mode")
        test_set = test_set[:len(running_log)]

    for data_item, label_item in tqdm(zip(running_log, test_set)):

        if turn >= 0:
            if data_item['turn_id'] != turn:
                continue

        n_total += 1

        completion = data_item['completion']

        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)  # a dictionary
        except:
            print("the output is not a valid SQL query")
            data_item['not_valid'] = 1
        predicted_slot_values = typo_fix(
            predicted_slot_values, ontology=ontology, version=args.mwz_ver)

        context_slot_values = data_item['last_slot_values']  # a dictionary

        # merge context and prediction
        if use_gold:
            all_slot_values = context_slot_values.copy()
        else:
            all_slot_values = prediction_recorder.state_retrieval(
                data_item).copy()

        for s, v in predicted_slot_values.items():
            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            elif v != "[DELETE]":
                all_slot_values[s] = v

        # some slots may contain multiple values
        all_slot_values = {k: v.split('|')[0]
                           for k, v in all_slot_values.items()}

        # record current turn prediction
        prediction_recorder.add_state(data_item, all_slot_values)

        # print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        # print(
        #     f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(
            all_slot_values, label_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1
        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
        else:
            result_dict[data_item['turn_id']].append(0)

    print(f"correct {n_correct}/{n_total}  =  {n_correct*100 / n_total}")
    print(f"Slot Acc {total_acc/n_total}")
    print(f"Joint F1 {total_f1/n_total}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")

    return


if __name__ == "__main__":

    # read the running log
    with open(args.run_log) as f:
        running_log = json.load(f)

    # read the testing file
    with open(args.test_fn) as f:
        test_set = json.load(f)

    eval(running_log, test_set, use_gold=False)
