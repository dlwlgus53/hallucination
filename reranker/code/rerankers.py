import pdb
from reranker.code.prompt import rerank_prompt


class Reranker():
    def __init__(self, model):
        self.model = model

    def rerank(self, embs, k):
        raise NotImplementedError


class LlamaReranker(Reranker):
    def __init__(self, model, overcheck, verbose=False, cot=False):
        super().__init__(model=model)
        self.overcheck = overcheck
        self.verbose = verbose
        self.cot = cot

    def rerank_best(self, examples, query, k):
        prompt = rerank_prompt(examples, query, k)

        while self.overcheck(prompt):
            examples = examples[:-1]
            if self.verbose:
                print("reduce")
            prompt = rerank_prompt(examples, query, k, self.cot)

        result = self.model(prompt)

        if self.verbose:
            print("prompt: ", prompt)
            print("result: ", result)
        if not self.cot:
            if len(result.strip()) == 0:
                return (examples[:k], 0)

            try:
                new_examples = []
                nums = result.replace("]", "").strip().split(",")
                for num in nums:
                    num = int(num.strip())
                    new_examples.append(examples[num-1])  # index is 1-based
                return (new_examples, 1)
            except:
                print("error result: ", result)
                return (examples[:k], 0)
        else:
            print("------------not to be here!!!!")
            return (examples[:k], 0)
