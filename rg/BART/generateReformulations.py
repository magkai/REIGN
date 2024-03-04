import json
import pickle
import sys
import tensorflow as tf
from transformers import TFBartForConditionalGeneration, BartTokenizer
import random
import threading
from multiprocessing import Process
from argparse import ArgumentParser

random.seed(7)
MAX_LENGTH = 50

parser = ArgumentParser()

parser.add_argument("--checkpointpath", dest="checkpoint_path")
parser.add_argument("--checknum", dest="checknum", type=int, default=3)
parser.add_argument("--filename", dest="filename")
parser.add_argument("--storepath", dest="store_path", default="")
parser.add_argument("--refoption", dest="ref_option", default="")
args = parser.parse_args()


data_type = args.data_type
file_name = args.filename
if args.store_path == "":
    store_path = file_name
else:
    store_path = args.store_path
checkpoint_path = args.checkpoint_path
check_num = args.checknum
ref_option = args.ref_option


model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
CAT_TOKENS = ["rc1", "rc2","rc3","rc4","rc5","rc6", "rc7","rc8","rc9", "rc10", "rc11", "rc12", "rc13", "rc14"]


ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(checkpoint_path + "/ckpt-" + str(check_num)).expect_partial()

with open(file_name , "rb") as infile:
    data_info = pickle.load(infile)



def generateReformulation(input):
    if not "input_ids" in input:
        tokenized_input =  tokenizer(input, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="tf")
    else:
        tokenized_input = input

    summary_ids = model.generate(tokenized_input["input_ids"], num_beams=5, num_return_sequences=1) 
    output = [

        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids
    ]

    return output

def generateOnePerSpecificCategories(questKeyList, catDict):
    count = 0
    for qId in questKeyList:
        if count%100 == 0:
            print("processed questions: ", count, flush=True)
        count +=1
        for cat in catDict[qId]:
            if int(cat) == 0:
                continue
            inData = data_info[qId]["conv_history"] + " " +  data_info[qId]["question"] + " " + CAT_TOKENS[int(cat)-1]
    
            gen_ref = generateReformulation(inData)
            newqid = qId[:-4] + "-" + str(cat) + "-" + str(0)
            data_info[newqid] = dict()
            data_info[newqid]["question"] = gen_ref[0]


def generateOnePerCategory(questKeyList):
    count = 0
    categories = list(range(1, 15))
    for qId in questKeyList:
        if count%100 == 0:
            print("processed questions: ", count, flush=True)
        count +=1
        for cat in categories:
            newqid = qId[:-4] + "-" + str(cat) + "-" + str(0)
            if newqid in data_info.keys():
                print("already contained: ", newqid)
                continue
            inData = data_info[qId]["conv_history"] + " " +  data_info[qId]["question"] + " " + CAT_TOKENS[int(cat)-1]
    
            gen_ref = generateReformulation(inData)
            
            data_info[newqid] = dict()
            data_info[newqid]["question"] = gen_ref[0]
            print("question: ", data_info[qId]["question"])
            print("ref: ", gen_ref[0])
            print("cat: ",  CAT_TOKENS[int(cat)-1])
            print("------------------------------------------", flush=True)




q_key_list = list(data_info.keys())

if ref_option == "onepercategory":
    generateOnePerCategory(q_key_list)
elif ref_option == "oneperspecifiedcat":
    catDict = dict()
    for qId in data_info.keys():
        if "available_cats" in data_info[qId].keys():
            catDict[qId] = data_info[qId]["available_cats"]
    generateOnePerSpecificCategories(q_key_list, catDict)


with open(store_path, "wb") as qfile:
    pickle.dump(data_info, qfile)

print("generation done")

