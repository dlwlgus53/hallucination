# here, make prompt for rerank
def rerank_prompt(items, query, k, cot):
    prompt = f"Given a set of dialgoue, please select the {k} most rerlated similar dialgoue with query.\n"

    for i, item in enumerate(items):

        prompt += f"{i+1}. sys': {item['dialog']['sys'][-1]}, 'usr': {item['dialog']['usr'][-1]}\n"

    prompt += f"query: 'sys' : {query['dialog']['sys'][-1]}, 'usr' : {query['dialog']['usr'][-1]}\n"
    if cot:
        prompt += f"Please select the {k} most contextually relevant dialogues for a given query, considering a specific reason.\n"
        prompt += "for example, ["
        prompt += "{'reason': 'Because both queries in the restaurant domain involve reservations for four people', 'index': 5}"
        prompt += "{'reason': 'Because booking a train ticket for a specific date', 'index': 7},"
        prompt += "{'reason': 'Because taxi service for a specific attraction', 'index': 10},"
        prompt += "]\n"
        prompt += "["
    else:
        prompt += f"Please select the {k} most related dialogue's with query in number of list format.\n"
        prompt += 'format is [number, number, number, ...]\n'
        prompt += '['
    return prompt
