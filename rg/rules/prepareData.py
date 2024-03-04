import json
import sys
import pickle
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("--inpath", dest="inpath")
parser.add_argument("--outpath", dest="outpath")
args = parser.parse_args()

with open(args.inpath, "r") as inFile:
    data = json.load(inFile)


def addRefCat(data):

    for entry in data:
        for q_entry in entry["questions"]:
            for ref_entry in q_entry["reformulations"]:
                ref_type = ref_entry["ref_type"]
                if ref_type == "INS_ENT":
                    ref_entry["ref_category"] = 1
                elif ref_type == "DEL_ENT":
                    ref_entry["ref_category"] = 2
                elif ref_type == "INS_PRED":
                    ref_entry["ref_category"] = 3
                elif ref_type == "DEL_PRED":
                    ref_entry["ref_category"] = 4
                elif ref_type == "INS_ENT_TYPE":
                    ref_entry["ref_category"] = 5
                elif ref_type == "DEL_ENT_TYPE":
                    ref_entry["ref_category"] = 6
                elif ref_type == "INS_ANS_TYPE":
                    ref_entry["ref_category"] = 7
                elif ref_type == "DEL_ANS_TYPE":
                    ref_entry["ref_category"] = 8
                elif ref_type == "SUBS_PRED":
                    ref_entry["ref_category"] = 9
                elif ref_type == "SUBS_ENT":
                    ref_entry["ref_category"] = 10
                elif ref_type == "SUBS_ENT_TYPE":
                    ref_entry["ref_category"] = 11
                elif ref_type == "SUBS_ANS_TYPE":
                    ref_entry["ref_category"] = 12
                elif ref_type == "SUBS_ENT_BY_TYPE":
                    ref_entry["ref_category"] = 13
                elif ref_type == "SUBS_ENT_BY_PRONOUN":
                    ref_entry["ref_category"] = 14

    return data


def processDataset(data, data_dict=None):
    if not data_dict:
        data_dict = dict()

    for entry in data:
        firstQuestion = ""
        prevQuestion = ""
        firstConv = ""
        prevConv = ""
        ref_prevQuestions = []
        ref_firstQuestions = []
        ref_firstConvs = []
        ref_prevConvs = []
        ref_prevQuestions_new = []
        ref_firstQuestions_new = []
        ref_firstConvs_new = []
        ref_prevConvs_new = [] 
        for i in range(15): 
            ref_prevQuestions.append("")
            ref_firstQuestions.append("")
            ref_prevConvs.append("")
            ref_firstConvs.append("")
            ref_prevQuestions_new.append("")
            ref_firstQuestions_new.append("")
            ref_prevConvs_new.append("")
            ref_firstConvs_new.append("")

        for q_entry in entry["questions"]:
            question = q_entry["question"]
            qId = q_entry["question_id"]
            turn = q_entry["turn"]
            dataId = qId + "-0-0"
            if not dataId in data_dict.keys():
                data_dict[dataId] = dict()
            data_dict[dataId]["ref_category"] = 0
            data_dict[dataId]["question"] = question
            data_dict[dataId]["answers"] = q_entry["answers"]
            answerString = ""
            aLen = 0
            for ans in q_entry["answers"]:
                aLen += 1
                if aLen == len(q_entry["answers"]):
                    answerString += ans["label"]
                else:
                    answerString += ans["label"] + "; "
            if turn == 0:
                data_dict[dataId]["question_history"] = ""
                data_dict[dataId]["conv_history"] = ""
                firstQuestion = question
                firstConv = question + " " + answerString
            elif turn == 1:
                data_dict[dataId]["question_history"] = prevQuestion
                data_dict[dataId]["conv_history"] = prevConv
            else:
                data_dict[dataId]["question_history"] = firstQuestion + " " +  prevQuestion
                data_dict[dataId]["conv_history"] = firstConv + ". " + prevConv
            
            ref_cat_num = []
           
            for i in range(15):
                ref_cat_num.append(-1)

           
            for ref_entry in q_entry["reformulations"]:
                ref_cat = ref_entry["ref_category"]
                ref_cat_num[ref_cat] += 1
                dataId = qId + "-" + str(ref_cat) + "-" + str(ref_cat_num[ref_cat])
                if not dataId in data_dict.keys():
                    data_dict[dataId] = dict()
                data_dict[dataId]["ref_category"] = ref_cat
                ref = ref_entry["reformulation"]
                data_dict[dataId]["question"] = ref
                data_dict[dataId]["answers"] = q_entry["answers"]
                if turn == 0:
                    data_dict[dataId]["question_history"] = ""
                    data_dict[dataId]["conv_history"] = ""
                    ref_firstQuestions_new[ref_cat] =  ref
                    ref_firstConvs_new[ref_cat] = ref + " " + answerString
                elif turn == 1:
                    if ref_prevQuestions[ref_cat] == "":
                        data_dict[dataId]["question_history"] = prevQuestion
                        data_dict[dataId]["conv_history"] = prevConv
                
                    else:
                        data_dict[dataId]["question_history"] = ref_prevQuestions[ref_cat]
                        data_dict[dataId]["conv_history"] = ref_prevConvs[ref_cat]
                else:
                    if ref_prevQuestions[ref_cat] == "":
                        prev = prevQuestion
                        prev_conv = prevConv
                    else:
                        prev = ref_prevQuestions[ref_cat]
                        prev_conv = ref_prevConvs[ref_cat]
                    if ref_firstQuestions[ref_cat] == "":
                        first = firstQuestion
                        first_conv = firstConv
                    else:
                        first = ref_firstQuestions[ref_cat]
                        first_conv = ref_firstConvs[ref_cat]
                    data_dict[dataId]["question_history"] = first + " " +  prev
                    data_dict[dataId]["conv_history"] = first_conv + ". " +  prev_conv
                ref_prevQuestions_new[ref_cat] = ref
                ref_prevConvs_new[ref_cat] = ref + " " + answerString

            prevQuestion = question
            prevConv = question + " " + answerString
            ref_firstQuestions = ref_firstQuestions_new.copy()
            ref_prevQuestions = ref_prevQuestions_new.copy()
            ref_firstConvs = ref_firstConvs_new.copy()
            ref_prevConvs = ref_prevConvs.copy()
    return data_dict 


def processDatasetWithoutRefs(data, data_dict=None):
    if not data_dict:
        data_dict = dict()

    for entry in data:
        firstQuestion = ""
        prevQuestion = ""
        firstConv = ""
        prevConv = ""

        for q_entry in entry["questions"]:
            question = q_entry["question"]
            qId = q_entry["question_id"]
            turn = q_entry["turn"]
            dataId = qId + "-0-0"
            if not dataId in data_dict.keys():
                data_dict[dataId] = dict()
            data_dict[dataId]["ref_category"] = 0
            data_dict[dataId]["question"] = question
            data_dict[dataId]["answers"] = q_entry["answers"]
            answerString = ""
            aLen = 0
            for ans in q_entry["answers"]:
                aLen += 1
                if aLen == len(q_entry["answers"]):
                    answerString += ans["label"]
                else:
                    answerString += ans["label"] + "; "
            if turn == 0:
                data_dict[dataId]["question_history"] = ""
                data_dict[dataId]["conv_history"] = ""
                firstQuestion = question
                firstConv = question + " " + answerString
            elif turn == 1:
                data_dict[dataId]["question_history"] = prevQuestion
                data_dict[dataId]["conv_history"] = prevConv
            else:
                data_dict[dataId]["question_history"] = firstQuestion + " " +  prevQuestion
                data_dict[dataId]["conv_history"] = firstConv + ". " + prevConv
            
         
            prevQuestion = question
            prevConv = question + " " + answerString
    
  
    return data_dict 


def isActionAllowed(struct_rep, refCat):
    possible = True
    #print("structrep0: ", struct_rep[0])
    if struct_rep[0] == 0:
        
        if refCat in [2,10,13,14]:
            possible = False
    else:
        if refCat == 1:
            possible = False
    
                
    if struct_rep[1] == 0:
        if refCat in [4,9]:
            possible = False
    else:
        if refCat == 3:
            possible = False

    if struct_rep[2] == 0:
        if refCat in [6,11]:
            possible = False
    else:
        if refCat == 5:
            possible = False

    if struct_rep[3] == 0:
        if refCat in [8,12]:
            possible = False
    else:
        if refCat == 7:
            possible = False

    return possible



def getStructuredRepresentation(data_dict):
    for entry in data:
        for q_entry in entry["questions"]:
            eId = q_entry["question_id"]
            qId = eId + "-0-0"
            print("qid: ", qId)
            if not qId in data_dict.keys():
                print("qId not contained")
                continue
            structuredRep = []
            for i in range(4):
                structuredRep.append(0)
            question = data_dict[qId]["question"]
            hist = data_dict[qId]["conv_history"]
            #check if conv entity in question
            for ent in q_entry["conv_entities"]:
                if "mention" in ent.keys() and ent["mention"] != "":
                    structuredRep[0] = 1
                    break
            #check if predicate in question
            if "question_predicate_mention" in q_entry["predicate"].keys() and len(q_entry["predicate"]["question_predicate_mention"]) > 0:
                structuredRep[1] = 1
            #check if entity type in question
            for ent in q_entry["entities"]:
                if "type_mention" in ent.keys() and ent["type_mention"] != "":
                    structuredRep[2] = 1
                    break
            #check if answer type in question
            if "ans_type_mention" in q_entry.keys():
                if q_entry["ans_type_mention"] != "":
                    structuredRep[3] = 1
           
            data_dict[qId]["structured_rep"] = structuredRep
            availAct = []
            for i in range(15):
                if isActionAllowed(structuredRep, i):
                    availAct.append(i)
            data_dict[qId]["available_cats"] = availAct
     
    


data = addRefCat(data)
data_new = processDataset(data)
getStructuredRepresentation(data_new)
print("done") 
#store train data
with open(args.outpath, "wb") as convFile:
    pickle.dump(data_new, convFile)

