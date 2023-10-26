import pdb
from reranker.code.prompt import rerank_prompt,  rerank_reasoning_prompt
import re
import time


class Reranker():
    def __init__(self, model):
        self.model = model

    def rerank(self, embs, k):
        raise NotImplementedError


def parsing_rerank_reasoning_result(result):
    success = 0
    index = -1
    reason = ""

    result = result.replace("\n", "").strip()
    result = result.lower()

    result = "index: " + result
    result = result.replace("[", "").replace("]", "").strip()
    result = result.replace(":", "").replace(",", "").strip()

    # Use a regular expression to find and extract the number
    index_match = re.search(r'(\d+)', result)

    if index_match:
        index = index_match.group(1)
        reason = re.sub(r'\d', '', result)
        reason = reason.replace("index", "").replace("reason", "").strip()
        success = 1
    else:
        print("Index not found in the response.")

    return success, int(index), reason


def parsing_rerank_result(result):
    success = 0
    index = -1
    reason = []  # always empty

    result = result.replace("\n", "").strip()
    result = result.lower()

    result = "index: " + result
    result = result.replace("[", "").replace("]", "").strip()
    result = result.replace(":", "").replace(",", "").strip()

    # Use a regular expression to find and extract the number
    index_match = re.search(r'(\d+)', result)

    if index_match:
        index = index_match.group(1)
        success = 1
    else:
        print("Index not found in the response.")

    return success, int(index), reason


class LlamaReranker(Reranker):
    def __init__(self, model, overcheck, cot):
        super().__init__(model=model)
        self.overcheck = overcheck
        self.cot = cot

    def rerank_best(self, examples, query, k):
        if self.cot:
            prompt_function = rerank_reasoning_prompt
            parsing_function = parsing_rerank_reasoning_result

        else:
            prompt_function = rerank_prompt
            parsing_function = parsing_rerank_result

        reasons = []
        re_examples = []
        re_examples_short = []
        indexs = []
        success = 1
        org_len = len(examples)
        org_examples = examples.copy()  # for debugging

        for i in range(k):
            prompt = prompt_function(examples, query)

            while self.overcheck(prompt):
                examples = examples[:-1]
                org_examples = org_examples[:-1]
                org_len = len(org_examples)
                prompt = prompt_function(examples, query)

            result = self.model(prompt, raw=1)

            success, index, reason = parsing_function(result)
            if not success:
                break
            else:
                reasons.append(reason)
                re_examples.append(examples[index-1])
                re_examples_short.append(
                    f"'sys' {examples[index-1]['dialog']['sys'][-1]} 'usr' {examples[index-1]['dialog']['usr'][-1]}")
                examples = examples[:index-1] + examples[index:]
                indexs.append(index)

                assert len(examples) == org_len - i - 1
                assert len(re_examples) == i + 1
                assert len(reasons) == i + 1

        if not success:
            re_examples = examples[:k]
        return re_examples, success, prompt+result+str(indexs), reasons, re_examples_short
