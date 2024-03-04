import json
import pickle
import re
import regex
import spacy
import copy
import sys
sys.path.append("utils")
from argparse import ArgumentParser

from CLOCQInterfaceClient import CLOCQInterfaceClient
clocq = CLOCQInterfaceClient(host="https://clocq.mpi-inf.mpg.de/api", port="443") 


parser = ArgumentParser()

parser.add_argument("--inpath", dest="inpath")
parser.add_argument("--annotateddataset", dest="annotated_dataset")
parser.add_argument("--outpath", dest="outpath")
args = parser.parse_args()


with open(args.inpath, "rb") as inFile:
    old_data_info = pickle.load(inFile)


with open(args.annotated_dataset, "r", encoding='utf-8') as convFile:
    annotated_dataset = json.load(convFile)


data_info = dict()
for qId in old_data_info.keys():
   # print("qid: ", qId)
    data_info[qId] = copy.deepcopy(old_data_info[qId])
    prevRefCat = qId.split("-")[2]
    origId = qId.split("-")[0] + "-" + qId.split("-")[1] + "-0-0"
    if qId == origId:
        continue
    if not origId in old_data_info.keys():
        print("not contained: ", origId)
        continue
    if prevRefCat == "1":
        newId = qId + "-2-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 2
    elif prevRefCat == "2":
        newId = qId + "-1-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 1
    elif prevRefCat == "3":
        newId = qId + "-4-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 4
    elif prevRefCat == "4":
        newId = qId + "-3-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 3
    elif prevRefCat == "5":
        newId = qId + "-6-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 6
    elif prevRefCat == "6":
        newId = qId + "-5-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 5
    elif prevRefCat == "7":
        newId = qId + "-8-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 8
    elif prevRefCat == "8":
        newId = qId + "-7-0"
        data_info[newId] = copy.deepcopy(data_info[origId])
        data_info[newId]["ref_category"] = 7


def treatPredicates(current_info, idToEntry):
    new_data_info = dict()
    for qId in current_info.keys():
        new_data_info[qId] = copy.deepcopy(current_info[qId])
        if not qId.endswith("-0-0"):
         #   print("continue: ", qId)
            continue

        origId = qId.split("-")[0] + "-" + qId.split("-")[1]
        if origId.split("-")[1] == "0":
            continue
        if not "question_predicate_mention" in idToEntry[origId]["predicate"].keys():
            print("no mention: ", origId)
            continue
       
        dropId =  origId + "-4-0"
        addId = origId + "-4-0-3-0"

        print("origid: ", origId)
        convHist = new_data_info[qId]["conv_history"]
        aLen = 0
        answerString = ""
        for ans in new_data_info[qId]["answers"]:
            aLen += 1
            if aLen == len(new_data_info[qId]["answers"]):
                answerString += ans["label"]
            else:
                answerString += ans["label"] + "; "
        if "." in convHist:
            convHist = convHist.split(".")[0] + ". " +  new_data_info[qId]["question"] + " " + answerString
        else:
            convHist = convHist +  ". " +  new_data_info[qId]["question"] + " " + answerString

        pmentions = idToEntry[origId]["predicate"]["question_predicate_mention"]
        newQuest = new_data_info[qId]["question"]
        for ment in pmentions:
            newQuest = newQuest.replace(ment, "")
        if len(newQuest)>2:
            new_data_info[dropId] = copy.deepcopy(new_data_info[qId])
            new_data_info[dropId]["conv_history"] = convHist
            new_data_info[dropId]["question"] = newQuest
            new_data_info[dropId]["ref_category"] = 4
            new_data_info[addId] = copy.deepcopy(new_data_info[qId])
            new_data_info[addId]["conv_history"] = convHist
            new_data_info[addId]["ref_category"] = 3
            

    return new_data_info



idToEntry = dict()
for entry in annotated_dataset:
    for q_entry in entry["questions"]:
        #new_entry = q_entry
        eId = q_entry["question_id"]
        idToEntry[eId] = q_entry


new_data_info = dict()

for qId in data_info.keys():
    new_data_info[qId] = copy.deepcopy(data_info[qId])
    if qId[-1] != "0":
        print("qid skip: ", qId)
        continue
    if qId.count("-")> 5:
        print("qid skip: ", qId)
        continue
    
    origId = qId.split("-")[0] + "-" +qId.split("-")[1]
    question = data_info[qId]["question"]
    if data_info[qId]["ref_category"] == 5:
        altTypes = []
        mention = ""
        for ent in idToEntry[origId]["entities"]:
            altTypes = []
            for type in ent["type"]:
               
                if type["label"] in question:
                    mention = type["label"]
                else:
                    altTypes.append(type["label"])
            if mention != "":
                break
        if mention == "":
            for ent in idToEntry[origId]["conv_entities"]:
                altTypes = []
                for type in ent["type"]:
                    if type["label"] in question:
                        mention = type["label"]
                    else:
                        altTypes.append(type["label"])
                if mention != "":
                    break
        if mention == "":
            print("no mention!!")
            continue
        for i in range(len(altTypes)):
            newId = qId + "-11-" + str(i)
            newTypeQuest = question
            newTypeQuest = regex.sub(mention,  altTypes[i] , newTypeQuest, flags=re.IGNORECASE)
            print("newTypeQuest 0: ", newTypeQuest)
            new_data_info[newId] = copy.deepcopy(new_data_info[qId])
            new_data_info[newId]["question"] = newTypeQuest
            new_data_info[newId]["ref_category"] = 11
        print("---------")
    elif data_info[qId]["ref_category"] == 1:    
        entInQuestion=""
        convEnt = False
        for ent in idToEntry[origId]["conv_entities"]:
            if ent["label"] in question:
                entInQuestion = ent
                convEnt = True
                break
        if entInQuestion == "":
            for ent in idToEntry[origId]["entities"]:
                altEnts = []
                if ent["label"] in question:
                    mention = ent["label"]
                    entInQuestion = ent
                    break
        if entInQuestion!= "":
            for i in range(len(entInQuestion["aliases"])):
                #we skip if contained..therefore starts at 10-1 and not 10-0!!
                newId = qId + "-10-" + str(i)
                if entInQuestion["aliases"][i] in question:
                    continue
                newQuest = question.replace(entInQuestion["label"], entInQuestion["aliases"][i])
                new_data_info[newId] = copy.deepcopy(new_data_info[qId])
                new_data_info[newId]["question"] = newQuest
                new_data_info[newId]["ref_category"] = 10
            if convEnt:
                for j in range(len(entInQuestion["type"])):
                    newId = qId + "-13-" + str(j)
                    if "type_mention" in entInQuestion.keys():
                        newQuest = regex.sub("(the )?"  +r'\b' +ent["label"] +r'\b', "",  question, flags=re.IGNORECASE)
                        new_data_info[newId] = copy.deepcopy(new_data_info[qId])
                        new_data_info[newId]["question"] = newQuest
                        new_data_info[newId]["ref_category"] = 13
                        continue
                   
                    newQuest = question.replace(ent["label"], entInQuestion["type"][j]["label"])
                    new_data_info[newId] = copy.deepcopy(new_data_info[qId])
                    new_data_info[newId]["question"] = newQuest
                    new_data_info[newId]["ref_category"] = 13
                
                newId = qId + "-14-" + str(0)
                if "type_mention" in entInQuestion.keys():
                    continue
                genderId = clocq.connect( entInQuestion["id"], "P21")
                if not genderId is None:
                    if len(genderId)==1:
                        if len(genderId[0])>2:
                            genderId = genderId[0][2]
                        else:
                            genderId = ""
                    else:
                        genderId = ""
                if genderId == "Q6581097":
                    pronoun = "he"
                elif genderId == "Q6581072":
                    pronoun = "she"
                else:
                    pronoun = "it"
                newQuest = regex.sub("(the )?"  +r'\b' +ent["label"] +r'\b', pronoun,  question, flags=re.IGNORECASE)
                new_data_info[newId] = copy.deepcopy(new_data_info[qId])
                new_data_info[newId]["question"] = newQuest
                new_data_info[newId]["ref_category"] = 14

        print("---------", flush=True)
    elif data_info[qId]["ref_category"] == 7:
        ansTypes = idToEntry[origId]["ans_types"]
        ansTypeInQuest = ""
        for ansType in ansTypes:
                if ansType.lower() in question.lower():
                    ansTypeInQuest = ansType
                    break
        if ansTypeInQuest != "":
            k = 0
            for ansType in ansTypes:
                if ansTypeInQuest == ansType:
                    continue
                newId = qId + "-12-" + str(k)
                k += 1
                newQuest = question.replace(ansTypeInQuest, ansType)
                new_data_info[newId] = copy.deepcopy(new_data_info[qId])
                new_data_info[newId]["question"] = newQuest
                new_data_info[newId]["ref_category"] = 12
        print("---------", flush=True)



new_data_info = treatPredicates(new_data_info, idToEntry)

ref_cat_num = []
for i in range(15):
    ref_cat_num.append(-1)
for qId in new_data_info.keys():
    print("QID: ", qId)
    if len(qId.split("-")) > 4:
        refcat = qId.split("-")[4]
    else:
        refcat = qId.split("-")[2]
    ref_cat_num[int(refcat)] += 1

print("ref cats: ", ref_cat_num) 

with open(args.outpath, "wb") as question_file:
    pickle.dump(new_data_info, question_file)
 
