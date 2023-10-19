# here, make prompt for rerank
def rerank_prompt(items, query, k, cot):
    prompt = f"Given a set of dialgoue, please select the {k} most rerlated similar dialgoue with query.\n"

    for i, item in enumerate(items):

        prompt += f"{i+1}. sys': {item['dialog']['sys'][-1]}, 'usr': {item['dialog']['usr'][-1]}\n"

    prompt += f"query: 'sys' : {query['dialog']['sys'][-1]}, 'usr' : {query['dialog']['usr'][-1]}\n"
    prompt += f"Please select the {k} most related dialogue's with query in number of list format.\n"
    if cot:
        prompt += 'with the detailed reason in list of dictionary type'
        prompt += 'for example, [{reason : ..., index : 6}, {reason : because.. , index : 13} ,,,]\n'
        prompt += "result : [{reason : "
    else:
        prompt += 'format is [number, number, number, ...]\n'
        prompt += '['
    return prompt
