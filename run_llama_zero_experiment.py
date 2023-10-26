import os
import json
import argparse
import copy
from collections import defaultdict
from tqdm import tqdm
from utils.helper import PreviousStateRecorder
from utils.typo_fix import typo_fix
from config import CONFIG
from utils.sql import sql_pred_parse, sv_dict_to_string
from prompting import get_prompt, conversion, table_prompt

from evaluate_metrics import evaluate
import init
import pdb
import logging

from saver import *
# input arguments


parser = argparse.ArgumentParser()
parser.add_argument('--model_version', type=str, help="name of the model",
                    required=True)  # e.g. "./data/mw21_10p_train_v3.json"
parser.add_argument('--train_fn', type=str, help="training data file (few-shot or full shot)",
                    required=True)  # e.g. "./data/mw21_10p_train_v3.json"
parser.add_argument('--retriever_dir', type=str, required=True,
                    help="sentence transformer model, npy saved path")
parser.add_argument('--retriever_model', type=str, required=False,
                    help="sentence transformer model saved path")
parser.add_argument('--output_dir', type=str, default="./expts/debug",
                    help="directory to save running log and configs")
parser.add_argument('--mwz_ver', type=str, default="2.1",
                    choices=['2.1', '2.4'], help="version of MultiWOZ")
parser.add_argument('--test_fn', type=str, default='',
                    help="file to evaluate on, empty means use the test set")
parser.add_argument('--num_examples', type=int, default=30,
                    help="number of examples to retrieve")
parser.add_argument('--num_re_examples', type=int, default=5,
                    help="number of examples to retrieve")
parser.add_argument('--short', type=int, default=0,
                    help="debugging mode")
parser.add_argument('--reranker', type=int, default=0,
                    help="use reranker or not")
parser.add_argument('--target_domain', type=str, default='hotel',
                    choices=['hotel', 'restaurant',
                             'attraction', 'taxi', 'train'],
                    help="target domain")

parser.add_argument('--verbose', type=int, default=0,
                    help="do you like verbose program?")
parser.add_argument('--cot', type=int, default=0,
                    help="cot prompting")
parser.add_argument('--seed', type=int, default=42,
                    help="cot prompting")
args = parser.parse_args()

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

init.init_experiment(args)
logger = logging.getLogger("my")

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)


# set up the completion function

# read the selection pool
with open(args.train_fn) as f:
    train_set = json.load(f)

# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
    if args.test_fn == "":
        test_set_path = "./data/mw21_100p_test.json"
else:
    ontology_path = CONFIG["ontology_24"]
    if args.test_fn == "":
        test_set_path = "./data/mw24_100p_test.json"

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

with open(ontology_path) as f:
    ontology = json.load(f)
with open(test_set_path) as f:
    test_sets = json.load(f)

# load the retriever


def run(test_set, turn=-1, use_gold=False):

    retriever = EmbeddingRetriever(datasets=[train_set],
                                   model_path=args.retriever_model if args.retriever_model else args.retriever_dir,
                                   search_index_filename=os.path.join(
                                   args.retriever_dir, "train_index.npy"),
                                   sampling_method="pre_assigned")

    if '7' in args.model_version:
        from llama_completion_7b import llama_check_over_length, llama_completion
    elif '13' in args.model_version:
        from llama_completion_13b import llama_check_over_length, llama_completion
    else:
        raise ValueError("model version not supported")

    complete_fn = llama_completion
    check_overlen_fn = llama_check_over_length

    # turn and use_gold are for analysis purpose
    # turn = -1 means evaluate all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evaluate two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    result_dict = defaultdict(list)  # use to record the accuracy
    Reranker = LlamaReranker(
        complete_fn, check_overlen_fn, cot=args.cot)
    selected_set = test_set
    # if needed, only evaluate on particular turns (analysis purpose)
    if turn >= 0:
        if not use_gold:
            raise ValueError(
                "can only evaluate particular turn when using gold context")
        selected_set = [d for d in test_set if len(
            d['dialog']['usr']) == turn + 1]

    prediction_recorder = PreviousStateRecorder()  # state recorder

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0
    if args.reranker:
        n_success = 0

    for data_item in tqdm(selected_set):
        if args.short != 0 and args.short == n_total:
            break
        n_total += 1

        completion = ""

        if use_gold:
            prompt_text = get_prompt(
                data_item, examples=retriever.item_to_nearest_examples(data_item, k=args.num_examples))
        else:
            predicted_context = prediction_recorder.state_retrieval(data_item)
            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context
            examples = retriever.item_to_nearest_examples(
                modified_item, k=args.num_examples)
            if args.reranker:  # use reranker or not
                examples, re_success, re_prompt, reasons, examples_short = Reranker.rerank_best(
                    examples=examples, query=modified_item, k=args.num_re_examples)

                n_success += re_success
                if args.verbose or n_total < 5:
                    logger.info(re_prompt)

            prompt_text = get_prompt(
                data_item, examples=examples, given_context=predicted_context)
        # logger.info the retrieved examples (without the sql table)
        if args.verbose or n_total < 5:
            logger.info(prompt_text.replace(conversion(table_prompt), ""))
        if re_success == 0:
            logger.warning("reranker failed")
            logger.warning(re_prompt)
            pdb.set_trace()

        # record the prompt
        data_item['prompt'] = prompt_text

        # completion
        overlen_flag = True
        while overlen_flag:
            prompt_text = get_prompt(
                data_item, examples=examples, given_context=predicted_context)
            overlen_flag = check_overlen_fn(prompt_text)

            # reduce the number of examples if overlength
            if overlen_flag:
                logger.info("prompt overlength")
                examples = examples[1:]

        completion = complete_fn(prompt_text)

        completion = conversion(completion, reverse=True)

        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)  # a dictionary
        except:
            logger.info("the output is not a valid SQL query")
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

        # record the predictions

        data_item['pred'] = all_slot_values
        data_item['ontology_path'] = ontology_path
        data_item['completion'] = completion
        if args.reranker:
            if args.cot:
                data_item['reranker_reasons'] = reasons
                data_item['reranker_prompt'] = re_prompt
                data_item['reranker_examples'] = examples_short

            data_item['reranker_success'] = re_success
        all_result.append(data_item)

        # logger.info the result
        if args.verbose or n_total < 5:
            logger.info(completion)
            logger.info(
                f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
            logger.info(
                f"pred turn change: {sv_dict_to_string(predicted_slot_values, sep='-')}")
            logger.info(
                f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}")
        # logger.info(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        # logger.info(
        # f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(
            all_slot_values, data_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
            # logger.info("\n=====================correct!=======================")
        else:
            result_dict[data_item['turn_id']].append(0)
            # logger.info("\n=====================wrong!=======================")

    score = {}
    logger.info(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}")
    score['JGA'] = n_correct
    score['n_total'] = n_total

    logger.info(f"Slot Acc {total_acc/n_total}")
    score['slot_acc'] = total_acc

    logger.info(f"Joint F1 {total_f1/n_total}")
    score['joint_f1'] = total_f1
    if args.reranker:
        logger.info(f"Success {n_success/n_total}")
        score['parsing_success'] = n_success

    # # calculate the accuracy of each turn
    # for k, v in result_dict.items():
    #     logger.info(
    #         f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")
    #     score[f"turn_{k}_acc"] = sum(v) / len(v)

    return score, all_result


def get_summed_scoring(scores_json):
    scores = defaultdict(float)
    for score in scores_json:
        scores['JGA'] += score['JGA']
        scores['n_total'] += score['n_total']
        scores['slot_acc'] += score['slot_acc']
        scores['joint_f1'] += score['joint_f1']
        if args.reranker:
            scores['parsing_success'] += score['parsing_success']

    scores['JGA'] /= scores['n_total']
    scores['slot_acc'] /= scores['n_total']
    scores['joint_f1'] /= scores['n_total']
    if args.reranker:
        scores['parsing_success'] /= scores['n_total']
    return scores


if __name__ == "__main__":

    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    test_sets = split_testset(test_sets, 20)
    from retriever.code.embed_based_retriever import EmbeddingRetriever
    from reranker.code.rerankers import LlamaReranker

    temp_folder = os.path.join(args.output_dir, 'logs')
    for idx, test_set in enumerate(test_sets):
        # if alerady done, skip
        if f"scores_{idx}.json" in os.listdir(temp_folder):
            logger.info(f"Skip {idx}'th result in {args.output_dir}")
            continue

        scores, all_results = run(test_set)

        with open(os.path.join(temp_folder, f"running_log_{idx}.json"), 'w') as f:
            json.dump(all_results, f, indent=4)

        with open(os.path.join(temp_folder, f"scores_{idx}.json"), 'w') as f:
            json.dump(scores, f, indent=4)

        logger.info(f"Save {idx}'th result in {temp_folder}")

    # add all the result together

    # load all files in the output folder
    all_results = []
    for fn in os.listdir(temp_folder):
        if fn.startswith("running_log"):
            with open(os.path.join(temp_folder, fn)) as f:
                all_results.extend(json.load(f))

    scores = []
    for fn in os.listdir(temp_folder):
        if fn.startswith("scores"):
            with open(os.path.join(temp_folder, fn)) as f:
                scores_ = json.load(f)
                scores.append(scores_)

    sumed_score = get_summed_scoring(scores)

    assert sumed_score['n_total'] == len(all_results)

    with open(os.path.join(args.output_dir, "running_log.json"), 'w') as f:
        json.dump(all_results, f, indent=4)

    with open(os.path.join(args.output_dir, "score.json"), 'w') as f:
        json.dump(sumed_score, f, indent=4)
