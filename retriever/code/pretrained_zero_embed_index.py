from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from tqdm import tqdm
import pdb 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--target_domain', type=str, required=True)
args = parser.parse_args()


# Embedding model

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
SAVE_NAME = f'all_mpnet_base_v2_zero_{args.target_domain}'


# ------ Configuration ends here ----------------


# path to save indexes and results
save_path = f"../expts/{SAVE_NAME}"
os.makedirs(save_path, exist_ok=True)

DEVICE = torch.device("cuda:0")
CLS_Flag = False

# Load model from HuggingFace Hub
print("Start load model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
print("Finish load model")

def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# function for embedding one string
def embed_single_sentence(sentence, cls=CLS_Flag):

    # Sentences we want sentence embeddings for
    sentences = [sentence]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(DEVICE)
    attention_mask = encoded_input['attention_mask'].to(DEVICE)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)

    # Perform pooling
    sentence_embeddings = None

    if cls:
        sentence_embeddings = model_output[0][:, 0, :]
    else:
        sentence_embeddings = mean_pooling(model_output, attention_mask)

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def read_MW_dataset(mw_json_fn, target_domain):
    '''
    example of history:
    '[system] great ! how about all saints church ? it has fantastic architecture and free entrance . 
    [user] could i get the address for it ? i would also like an expensive place to eat around it .'
    '''

    # only care domain in test
    DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']
    DOMAINS.remove(target_domain)
    
    assert len(DOMAINS) == 4
    
    with open(mw_json_fn, 'r') as f:
        data = json.load(f)

    dial_dict = {}

    for turn in tqdm(data):
        # filter the domains that not belongs to the test domain
        if not set(turn["domains"]).issubset(set(DOMAINS)):
            continue

        # update dialogue history
        sys_utt = turn["dialog"]['sys'][-1]
        usr_utt = turn["dialog"]['usr'][-1]

        if sys_utt == 'none':
            sys_utt = ''
        if usr_utt == 'none':
            usr_utt = ''

        history = f"[system] {sys_utt} [user] {usr_utt}"

        # store the history in dictionary
        name = f"{turn['ID']}_turn_{turn['turn_id']}"
        dial_dict[name] = history
        
    print('len of dict : ', len(dial_dict))
    
    return dial_dict


mw_train = read_MW_dataset("../../data/mw21_100p_train.json", args.target_domain)
mw_dev = read_MW_dataset("../../data/mw21_100p_dev.json", args.target_domain)
mw_test = read_MW_dataset("../../data/mw21_100p_test.json", args.target_domain)
print("Finish reading data")


def store_embed(input_dataset, output_filename, forward_fn):
    outputs = {}
    with torch.no_grad():
        for k, v in tqdm(input_dataset.items()):
            outputs[k] = forward_fn(v).detach().cpu().numpy()
    np.save(output_filename, outputs)
    return


# store the embeddings
store_embed(mw_train, f"{save_path}/mw21_train_{SAVE_NAME}.npy",
            embed_single_sentence)
store_embed(mw_dev, f"{save_path}/mw21_dev_{SAVE_NAME}.npy",
            embed_single_sentence)
store_embed(mw_test, f"{save_path}/mw21_test_{SAVE_NAME}.npy",
            embed_single_sentence)
print("Finish Embedding data")
