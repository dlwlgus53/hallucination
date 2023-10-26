# here, make prompt for rerank
import pdb


# def rerank_prompt(items, query, k):
#     prompt = f"Given a set of dialgoue, please select the {k} most rerlated similar dialgoue with query.\n"
#     for i, item in enumerate(items):

#         prompt += f"{i+1}. 'sys': {item['dialog']['sys'][-1]}, 'usr': {item['dialog']['usr'][-1]}\n"

#     prompt += f"query: 'sys' : {query['dialog']['sys'][-1]}, 'usr' : {query['dialog']['usr'][-1]}\n"

#     prompt += f"Please select the {k} most related dialogue's with query in number of list format.\n"
#     prompt += 'format is [number, number, number, ...]\n'
#     prompt += '['
#     return prompt


def rerank_prompt(items, query):
    prompt = f"Given a set of dialgoue, please select the most rerlated similar dialgoue with query.\n"
    for i, item in enumerate(items):

        prompt += f"{i+1}. 'sys': {item['dialog']['sys'][-1]}, 'usr': {item['dialog']['usr'][-1]}\n"

    prompt += f"query: 'sys' : {query['dialog']['sys'][-1]}, 'usr' : {query['dialog']['usr'][-1]}\n"

    prompt += f"Please select the  most related dialogue's with query.\n"
    prompt += f"index can be 1 to {len(items) } \n"
    prompt += 'index : '
    return prompt


def rerank_reasoning_prompt(items, query):
    prompt = f"Given a set of dialgoue, please select the most rerlated similar dialgoue with query.\n"
    for i, item in enumerate(items):
        prompt += f"{i+1}. 'sys': {item['dialog']['sys'][-1]}, 'usr': {item['dialog']['usr'][-1]}\n"

    prompt += f"query: 'sys' : {query['dialog']['sys'][-1]}, 'usr' : {query['dialog']['usr'][-1]}\n"

    prompt += f"In your response, please provide the index of the most pertinent dialogue and a concise explanation for your selection. \n"
    prompt += "Use the following format Index: [Index number], Reason: [Brief explanation]. \n"
    prompt += "Fill in the [Index number] with the relevant index and [Brief explanation] with your rationale. \n"
    prompt += f"index can be 1 to {len(items) } \n"

    prompt += "Index: "
    return prompt


'''
def reasoning_prompt(item, query):
    prompt = f"Explain why the two dialogue is similar, neatly, in  one sentence\n"
    prompt += f"<Dialogue1> 'sys' : {query['dialog']['sys'][-1]}, 'usr' : {query['dialog']['usr'][-1]}\n"
    prompt += f"<Dialogue2> 'sys': {item['dialog']['sys'][-1]}, 'usr': {item['dialog']['usr'][-1]}\n"
    prompt += f"Explanation : Both dialogues are similar because"
    return prompt
'''
