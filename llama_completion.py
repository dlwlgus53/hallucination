from transformers import pipeline, AutoTokenizer
import torch
import pdb


class Llama_Generation:
    def __init__(self, model_path, max_new_length=512, max_length=4096, device=0):
        print(f"start load {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generator = pipeline(
            'text-generation', model=model_path, device=device)
        self.max_length = max_length
        self.max_new_length = max_new_length
        print(f"finish load {model_path}")

    def llama_check_over_length(self, prompt, report_len=False):
        input_ids = tokenizer(prompt)['input_ids']
        if report_len:
            print(f"length is {len(input_ids)}")
        return len(input_ids) > self.max_length-self.max_new_length

    def llama_completion(self, prompt, raw=0):
        with torch.no_grad():
            generated_text = self.generator(prompt, do_sample=True,
                                            early_stopping=True,
                                            max_new_tokens=self.max_new_length,
                                            temperature=1e-10,
                                            num_return_sequences=1)[0]['generated_text']
        generated_text = generated_text.replace(prompt, "")

        if raw:
            return generated_text

        stop = ['--', '\n', ';', '#']
        stop_index = len(generated_text)
        # TODO is it ok?
        for i, c in enumerate(generated_text):
            if c in stop:
                stop_index = i
                break
        return generated_text[:stop_index]


if __name__ == "__main__":
    print("?")
    model = Llama_Generation('gpt2', max_new_length=10)
    print(model.llama_completion("hello", raw=1))
    print(model.llama_completion("hello", raw=0))
