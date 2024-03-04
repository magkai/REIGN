import json
import pickle
import threading
import sys
import os
sys.path.append("utils")
import utils as ut
from QAEncoding import QAEncoder
from Neo4jDatabaseConnection import KGEnvironment
import concurrent.futures
import threading
import random
from argparse import ArgumentParser


random.seed(7)

parser = ArgumentParser()

parser.add_argument("--filename", dest="filename")
parser.add_argument("--storepath", dest="store_path", default="")
parser.add_argument("--entityfilename", dest="ent_file_name", default="")
parser.add_argument("--datatype", dest="data_type", default="train")
parser.add_argument("--category", dest="cat_num")
parser.add_argument("--uri", dest="uri")
args = parser.parse_args()

data_type = args.data_type
file_name = args.filename
if args.store_path == "":
    store_path = file_name
else:
    store_path = args.store_path

cat_num = args.cat_num
ent_file_name = args.ent_file_name

with open(file_name , "rb") as infile:
    data_info = pickle.load(infile)

if os.path.exists(ent_file_name):
    with open(ent_file_name , "rb") as outfile:
        ent_info  = pickle.load(outfile)
else:
    ent_info = dict()



qaEnc = QAEncoder()
kgEnv = KGEnvironment(args.uri)


new_ent_info = dict()

for qId in data_info.keys():
    for ent in data_info[qId]["entities"]:
        if  ent in ent_info.keys():
            new_ent_info[ent] = ent_info[ent]
        else:
            print("not contained: ", ent)
            new_ent_info[ent] = dict()
            new_ent_info[ent]["actions"] = []
            new_ent_info[ent]["encoded_actions"] = []
            new_ent_info[ent]["action_num"] = 0
    

for ent in new_ent_info.keys():

    if len(new_ent_info[ent]["actions"]) == 0:
        new_paths = kgEnv.get_one_hop_nodes(ent) #paths["Q6256"]#
        new_ent_info[ent]["actions"] = new_paths
             
        if len(new_paths) == 0:
            print("empty entry for entity: ", ent)
            continue
      
        enc_acts, action_num = qaEnc.getActionEncodings(qaEnc.getActionLabels(new_paths)) 
        new_ent_info[ent]["encoded_actions"] = enc_acts
        new_ent_info[ent]["action_num"] = action_num

with open(ent_file_name , "wb") as outfile:
   pickle.dump(new_ent_info, outfile)


#get only reachable entities for training
number_of_training_samples = 0
number_of_samples_answer_reachable = 0
if data_type == "train":
    print("get reachable")
    for qid in data_info.keys():
        reachEnts = []
        for ent in data_info[qid]["entities"]:
            number_of_training_samples += 1
            for path in new_ent_info[ent]["actions"]:
                #print("path: ", path)
                pathend = path[2] 
                if ut.is_timestamp(pathend):
                    pathend = ut.convertTimestamp(pathend)
                if qid.endswith("-0-0"):
                    orig_qId = qid
                else:
                    splitted = qid.split("-")
                    orig_qId = splitted[0] + "-" + splitted[1] + "-0-0"
                answers =  [ans["id"] for ans in data_info[orig_qId]["answers"]]
                if pathend in answers:
                    number_of_samples_answer_reachable += 1
                    reachEnts.append(ent)
                    break
        data_info[qid]["entities"] = reachEnts

    with open(store_path , "wb") as outfile:
        pickle.dump(data_info, outfile)

    print("reachable entities stored")