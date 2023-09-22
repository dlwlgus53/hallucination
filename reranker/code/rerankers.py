import pdb
from reranker.code.prompt import rerank_prompt


class Reranker():
    def __init__(self, model):
        self.model = model

    def rerank(self, embs, k):
        raise NotImplementedError


class LlamaReranker(Reranker):
    def __init__(self, model, overcheck):
        super().__init__(model=model)
        self.overcheck = overcheck

    def rerank_best(self, examples, query, k):
        prompt = rerank_prompt(examples, query, k)

        while self.overcheck(prompt):
            examples = examples[:-1]
            print("reduce")
            prompt = rerank_prompt(examples, query, k)

        result = self.model(prompt)
        if len(result.strip()) == 0:
            return (examples[:k], 0)
        else:
            try:
                new_examples = []
                nums = result.replace("]", "").strip().split(",")
                for num in nums:
                    num = int(num.strip())
                    new_examples.append(examples[num])
                return (new_examples, 1)
            except:
                print("error result: ", result)
                return (examples[:k], 0)
