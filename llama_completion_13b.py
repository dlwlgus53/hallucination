from transformers import pipeline, AutoTokenizer
import torch
import pdb
device = -1
if torch.cuda.is_available():
    device = 0

MAX_NEW_LENGTH = 512
MAX_LENGTH = 4096

print("start load 13b model")
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
generator = pipeline(
    'text-generation', model='meta-llama/Llama-2-13b-chat-hf', device=device)
print("finish load 13b  model")

# if the input is over maximum length, return True


def llama_check_over_length(prompt, report_len=False):
    input_ids = tokenizer(prompt)['input_ids']
    if report_len:
        print(f"length is {len(input_ids)}")
    return len(input_ids) > MAX_LENGTH-MAX_NEW_LENGTH


def llama_completion(prompt, raw=0):
    with torch.no_grad():
        generated_text = generator(prompt, do_sample=True,
                                   max_new_tokens=MAX_NEW_LENGTH,
                                   temperature=1e-10,
                                   num_return_sequences=1)[0]['generated_text']
    generated_text = generated_text.replace(prompt, "")
    if raw:
        return generated_text
    stop = ['--', '\n', ';', '#']
    stop_index = len(generated_text)
    for i, c in enumerate(generated_text):
        if c in stop:
            stop_index = i
            break
    return generated_text[:stop_index]
