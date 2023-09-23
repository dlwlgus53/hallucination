python retriever_finetuning.py \
--train_fn ../../data/mw21_5p_train_v2.json \
--save_name 5p_test \
--epoch 0 \
--topk 10 \
--toprange 200


# for the pretraining model
python pretrained_embed_index.py

# for the zeroshot pretrained model
python pretrained_zero_embed_index.py
