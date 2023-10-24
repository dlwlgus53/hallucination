import pdb
from reranker.code.prompt import rerank_prompt


class Reranker():
    def __init__(self, model):
        self.model = model

    def rerank(self, embs, k):
        raise NotImplementedError


class LlamaReranker(Reranker):
    def __init__(self, model, overcheck, cot):
        super().__init__(model=model)
        self.overcheck = overcheck
        self.cot = cot

    def rerank_best(self, examples, query, k):
        reasons = []
        prompt = rerank_prompt(examples, query, k, self.cot)
        while self.overcheck(prompt):
            examples = examples[:-1]
            prompt = rerank_prompt(examples, query, k, self.cot)

        result = self.model(prompt, raw=1)
        result = result[:result.find("]")+1]
        re_examples = []
        success = 0
        if not self.cot:
            if len(result.strip()) == 0:
                re_examples, success = examples[:k], 0
            try:
                nums = result.replace("]", "").strip().split(",")
                for num in nums:
                    num = int(num.strip())
                    re_examples.append(examples[num-1])  # index is 1-based
                success = 1
            except:
                print("error result: ", result)
                re_examples, success = examples[:k], 0

        else:
            try:

                result = "[" + result.strip()
                result = result.replace(".", "")
                result_items = result.split("}")
                for item in result_items:
                    if "reason" in item and "index" in item:
                        reason = item[item.find(
                            "reason") + len("reason"): item.find("index")]
                        reason = reason.replace("'", "").replace(
                            '"', "").replace(":", "").replace(",", "").strip()

                        index = item[item.find("index") + len("index"):]
                        index = index.replace("'", "").replace(
                            '"', "").replace(":", "").replace(",", "").strip()

                        reasons.append(reason)
                        re_examples.append(examples[int(index)-1])

                    success = 1
                    # case2:  error in example number

                if len(re_examples) != k:
                    re_examples, success = examples[:k], 0
            except Exception as e:
                # case 3 : unknown error
                print(e)
                print("error result: ", result)

                re_examples, success = examples[:k], 0  # TODO : chnge here
        return re_examples[:k], success, prompt+result, reasons
