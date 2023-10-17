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
        prompt = rerank_prompt(examples, query, k, self.cot)
        while self.overcheck(prompt):
            examples = examples[:-1]
            prompt = rerank_prompt(examples, query, k, self.cot)

        result = self.model(prompt)

        re_examples = []
        success = 0
        if not self.cot:
            if len(result.strip()) == 0:
                re_examples, success = examples[:k], 0

            try:
                new_examples = []

                nums = result.replace("]", "").strip().split(",")
                for num in nums:
                    num = int(num.strip())
                    new_examples.append(examples[num-1])  # index is 1-based
                re_examples, success = new_examples, 1
            except:
                print("error result: ", result)
                re_examples, success = examples[:k], 0
        else:
            try:
                "[{'reason' : something, 'index' : something}, {reason : something, 'index' : something], ...]"
                new_examples = []
                result = '[{reason : '+result.strip()
                result_list = result.split("index : ")[1:]
                for item in result_list:
                    index = int(item.split("}")[0].strip())
                    new_examples.append(examples[index-1])
                re_examples, success = new_examples, 1
            except:
                print("error result: ", result)
                re_examples, success = examples[:k], 0  # TODO : chnge here

        return re_examples[:k], success, prompt+'\n'+result
