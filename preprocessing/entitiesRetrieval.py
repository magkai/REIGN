
import json
import pickle
import sys


import elq.main_dense as main_dense
import argparse
import numpy as np
#from argparse import ArgumentParser


models_path = "models/" # the path where you stored the ELQ models

config = {
    "interactive": False,
    "biencoder_model": models_path+"elq_wiki_large.bin",
    "biencoder_config": models_path+"elq_large_params.txt",
    "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "output_path": "logs/", # logging directory
    "faiss_index": "hnsw",
    "index_path": models_path+"faiss_hnsw_index.pkl",
    "num_cand_mentions": 10,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    "threshold": -4.5,
    "eval_batch_size": 10
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

id2wikidata = json.load(open("models/id2wikidata.json"))
parser = argparse.ArgumentParser()

parser.add_argument("--filename", dest="filename")
parser.add_argument("--questioncache", dest="question_cache", default="")
parser.add_argument("--cachestorepath", dest="cache_store_path", default="")
parser.add_argument("--storepath", dest="store_path", default="")
parser.add_argument("--datatype", dest="data_type", default="train")
myParserArgs = parser.parse_args()

data_type = myParserArgs.data_type
file_name = myParserArgs.filename
if myParserArgs.store_path == "":
    store_path = file_name
else:
    store_path = myParserArgs.store_path

with open(file_name , "rb") as infile:
    data_info = pickle.load(infile)

if myParserArgs.question_cache == "":
    question_cache = dict()
else:
    with open(myParserArgs.question_cache , "rb") as infile:
        question_cache = pickle.load(infile)

def doLinking():

    data_to_link = []

    for qId in data_info.keys():
        #if data_info[qId]["ref_category"] == 13 or data_info[qId]["ref_category"] == 14:

        data_info[qId]["entities"] = []
        if data_info[qId]["ref_category"]>2 and data_info[qId]["ref_category"]<13:
            continue
        
        #convId = qId.split("-")[1]
        data_to_link.append({"id": qId , "text": data_info[qId]["question_history"] + " " +  data_info[qId]["question"]})
    # data_to_link.append({"id": qId , "text": convHistory[qId] + refs[qId]})
    
        
    predictions = main_dense.run(args, None, *models, test_data=data_to_link)

    with open("../data/conv_mix/" + data_type + "/elq_predictions_" + data_type + "set.pickle" , "wb") as outfile:
        pickle.dump(predictions, outfile)

    for prediction in predictions:

        pred_ids = [id2wikidata.get(wikipedia_id) for (wikipedia_id, a, b) in prediction['pred_triples']]
        for predId in pred_ids:
            if predId is None:
                continue
            if not predId in data_info[prediction['id']]["entities"]:
            
                data_info[prediction['id']]["entities"].append(predId)

    return 


def doLinkingForRefs():
    data_to_link = []

    for qId in data_info.keys():
        if "entities" in data_info[qId].keys():
            print("entities already available for ", qId)
            continue
        data_info[qId]["entities"] = []

        if qId.endswith("-0-0"):
            orig_qId = qId
        else:
            splitted = qId.split("-")
            orig_qId = splitted[0] + "-" + splitted[1] + "-0-0"
        found = False
        if orig_qId in question_cache.keys():
            for i in range(len(question_cache[orig_qId]["questions"])):
                if data_info[qId]["question"].lower() == question_cache[orig_qId]["questions"][i].lower():
                    data_info[qId]["entities"] =  question_cache[orig_qId]["entities"][i]
                    found = True
                    print("question found in cache: ", data_info[qId]["question"], ", qid: ", qId)
                    break

   #     if "ref_catgeory" in data_info[qId].keys():
    #        if data_info[qId]["ref_category"]>2 and data_info[qId]["ref_category"]<13:
     #           continue
        if not found:
            data_to_link.append({"id": qId , "text": data_info[orig_qId]["question_history"] + " " +  data_info[qId]["question"]})

    predictions = []
    if len(data_to_link)>0:    
        predictions = main_dense.run(args, None, *models, test_data=data_to_link)


    for prediction in predictions:
        if prediction['id'].endswith("-0-0"):
            orig_qId = prediction['id']
        else:
            splitted = prediction['id'].split("-")
            orig_qId = splitted[0] + "-" + splitted[1] + "-0-0"
        pred_ids = [id2wikidata.get(wikipedia_id) for (wikipedia_id, a, b) in prediction['pred_triples']]
        for predId in pred_ids:
            if predId is None:
                continue
            if not predId in data_info[prediction['id']]["entities"]:
            
                data_info[prediction['id']]["entities"].append(predId)
        if orig_qId in question_cache.keys():
            question_cache[orig_qId]["questions"].append( data_info[prediction['id']]["question"])
            question_cache[orig_qId]["entities"].append(data_info[prediction['id']]["entities"])
        else:
            question_cache[orig_qId] = dict()
            question_cache[orig_qId]["questions"] = [data_info[prediction['id']]["question"]]
            question_cache[orig_qId]["entities"] = [data_info[prediction['id']]["entities"]]
    return 
    

def addEntsForRefCategories():   
    for refId in data_info.keys():
        if data_info[refId]["ref_category"]>2 and data_info[refId]["ref_category"]<13:
            split = refId.split("-")
            questionId = str(split[0]) + "-" + str(split[1]) + "-0-0"
            data_info[refId]["entities"] = data_info[questionId]["entities"].copy()


def addAnswerEntities():
    for qId in data_info.keys():
        splitted_id = qId.split("-")
        quest_id = splitted_id[0] + "-" + splitted_id[1]
        #print("quest_id: ", quest_id)
        turn = int( splitted_id[1])
        hist_ans = []
        for i in range(turn):
            hist_id = splitted_id[0] + "-" + str(i) + "-0-0"
            answerIds = [ans["id"] for ans in data_info[hist_id]["answers"] ] 
         #   print("histid: ", hist_id, "answerid: ", answerIds)
            for aId in answerIds:
                if aId.startswith("Q"):
                    hist_ans.append(aId)
        #print("hist ans: ", hist_ans)
        #print("current ents bef: ", data_info[qId]["entities"])
        if len(hist_ans) == 0:
            continue
        data_info[qId]["entities"].extend(hist_ans)
        allEnts = list(set(data_info[qId]["entities"]))
        data_info[qId]["entities"] = allEnts
    return

def addEntitiesFromDataset():
    for refId in data_info.keys():
       
        split = refId.split("-")
        
        questionId = str(split[0]) + "-" + str(split[1])
      
        entCandidates = []
        for entry in data:
            for q_entry in entry["questions"]:
                if q_entry["question_id"] == questionId:
                    entCandidates = [ent["id"] for ent in q_entry["entities"]]
                    break
      
        for ent in entCandidates:
            if not ent in data_info[refId]["entities"]:
                data_info[refId]["entities"].append(ent)


doLinkingForRefs()

if data_type == "train": #or data_type == "dev":
    print("add answers!!!")
    addAnswerEntities()


with open(store_path , "wb") as outfile:
    pickle.dump(data_info, outfile)

if myParserArgs.cache_store_path != "":
    with open(myParserArgs.cache_store_path , "wb") as outfile:
        pickle.dump(question_cache, outfile)

print("elq done", flush=True)



