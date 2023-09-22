# here, make prompt for rerank
def rerank_prompt(items, query, k):
    prompt = f"Given a set of dialgoue, please select the {k} most rerlated similar dialgoue with query.\n"

    for i, item in enumerate(items):
        prompt += f"{i+1}. sys': {item['dialog']['sys'][0]}, 'usr': {item['dialog']['usr'][0]}\n"

    prompt += f"query: 'sys' : {query['dialog']['sys'][0]}, 'usr' : {query['dialog']['usr'][0]}\n"
    prompt += f"Please select the {k} most related dialogue with query in list type.\n"
    prompt += f"result : ["

    return prompt
