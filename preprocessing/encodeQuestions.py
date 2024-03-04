import json
import pickle
import sys

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from argparse import ArgumentParser


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)

parser = ArgumentParser()

parser.add_argument("--filename", dest="filename")
parser.add_argument("--encodehistory", action='store_true')
parser.add_argument("--storepath", dest="store_path", default="")
parser.add_argument("--datatype", dest="data_type", default="train")
args = parser.parse_args()

data_type = args.data_type
file_name = args.filename
if args.store_path == "":
    store_path = file_name
else:
    store_path = args.store_path

print("history: ", args.encodehistory)

with open(file_name , "rb") as infile:
    data_info = pickle.load(infile)


#get pre-trained BERT embeddings for question
def encodeQuestions(questions):
    tokenized_input =  tokenizer(questions, return_tensors="tf", padding=True) 
    encoded_input = encoder(tokenized_input, output_hidden_states=True)
    #take average over all hidden layers
    all_layers = [ encoded_input.hidden_states[l] for l in range(1,13)]
    encoder_layer = tf.concat(all_layers, 1)
    pooled_output = tf.reduce_mean(encoder_layer, axis=1)

    return pooled_output


for key in data_info.keys():
    if "encoded_question" in data_info[key].keys():
        continue
    question = data_info[key]["question"]
    data_info[key]["encoded_question"] = encodeQuestions(question)

if args.encodehistory:
    for key in data_info.keys():
        if "encoded_question_history" in data_info[key].keys():
            continue
        question = data_info[key]["question"]
        origId = key.split("-")[0] + "-" + key.split("-")[1] + "-0-0"
        history = data_info[origId]["conv_history"]
        data_info[key]["encoded_question_history"] = encodeQuestions(history + " " + question)


with open(store_path , "wb") as outfile:
    pickle.dump(data_info, outfile)
