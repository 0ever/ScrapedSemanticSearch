import os
import requests
import csv
import pandas as pd
import torch
import faiss
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from flask import Flask, jsonify, request
from collections import Counter
os.environ['KMP_DUPLICATE_LIB_OK']='True'

main = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained('ddobokki/electra-small-nli-sts')
model = AutoModel.from_pretrained('ddobokki/electra-small-nli-sts', from_pretrained = True)

with open('/Users/ignas/desktop/dl3/data/faiss_mean_250_v5.csv','r') as result:
  df = pd.read_csv(result)
embeddings_dataset = Dataset.from_pandas(df)
embeddings_dataset.load_faiss_index(index_name='embeddings', file='/Users/ignas/desktop/dl3/data/faiss_index_v5')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(text_list):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length = 256)
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

def get_question_embedding(question):
    with torch.no_grad():
        question_embedding = get_embeddings([question]).numpy()
    return question_embedding

def get_result_list(question):
    res_list = []
    q = get_question_embedding(question)
    scores, samples = embeddings_dataset.get_nearest_examples('embeddings', q, k=5)
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    for _, row in samples_df.iterrows():
        res_dict = {'HEADING' : row.Heading, 'BODY' : row.Body, 'SCORES' : str(row.scores)}
        res_list.append(res_dict)
    return res_list

@main.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        print(data.get('text'))
        res = get_result_list(data.get('text'))
    return jsonify(res)


if __name__ == '__main__':
    main.run()
