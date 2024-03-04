import json
import pickle
import copy
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--inputfile", dest="inputfile")
parser.add_argument("--outputfile", dest="outputfile")
args = parser.parse_args()

with open(args.inputfile, "rb") as infile:
    data_info = pickle.load(infile)

def transformIntoConversations(data_info):
    new_data_info = dict()
    for qId in data_info.keys():
        convId = qId.split("-")[0] + "-" + qId.split("-")[1]
        if not convId in new_data_info.keys():
            new_data_info[convId] = dict()
            new_data_info[convId]['conversation_id'] = convId
            new_data_info[convId]['questions'] = []

        new_turn = dict()
        origId = qId.split("-")[0] + "-" + qId.split("-")[1] + "-0-0"
        new_turn["question"] = data_info[qId]["question"]
        new_turn["answers"] = data_info[origId]["answers"]
        turn = int(qId.split("-")[1])
        new_turn["turn"] = turn
        new_turn['question_id'] = qId

        hist = []
        for i in range(turn):
            prevId = qId.split("-")[0] + "-" + str(i) + "-0-0"
            hist.append(data_info[prevId]["question"])
            hist.append(", ".join([answer["label"]
                        for answer in data_info[prevId]["answers"]]))

        new_turn["history"] = " ||| ".join(hist)
    # print("new info; ", new_data_info[qId])
        # print("-------------")
        new_data_info[convId]['questions'].append(new_turn)

    new_data_list = list(new_data_info.values())
    with open(args.outputfile, "w") as outFile:
        json.dump(new_data_list, outFile)
    return


transformIntoConversations(data_info)
