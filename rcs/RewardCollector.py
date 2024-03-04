import json
import pickle
import copy
from argparse import ArgumentParser


class RewardCollector():

    def __init__(self, data_info, rewardType, outputFile, absolute, reversed):

        self.data_info = data_info
        self.rewardType = rewardType
        self.outputFile = outputFile
        self.absolute = absolute
        self.reversed = reversed
        self.rewardDict = dict()

        if self.rewardType == "rrd":
            self.getReciprocalRankReward()
        elif self.rewardType == "mpd":
            self.getPredictionProbabilityReward()
        else:
            print("unknown reward type")

        
        self.storeRewardInfo()


    def getReciprocalRankReward(self):
        
        origResult = dict()
        for conv in self.data_info:
            for turn in conv["questions"]:
                qId = turn['question_id']
                if not qId.endswith("-0-0"):
                    continue
                rr = -1.0
                for preda in turn["ranked_answers"]: 
                    if preda['rank'] > 5:
                        break
                    if preda['answer']['id'] in [a['id'] for a in turn['answers']]:
                        rr = 1.0/preda['rank']
                        break

                origResult[qId] = rr
        for conv in self.data_info:
            for turn in conv["questions"]:
                rVal = -1.0
                for preda in turn["ranked_answers"]:
                    if preda['rank'] > 5:
                        break
                    if preda['answer']['id'] in [a['id'] for a in turn['answers']]:
                        rVal = 1.0/preda['rank']
                        break

                qId = turn['question_id']
                convId = qId.split("-")[0] + "-" + qId.split("-")[1]
                if self.absolute:
                    if self.reversed:
                        reward = 1.0 -rVal
                    else:
                        reward = rVal
                else:
                    origId = convId + "-0-0"
                    if origId in origResult.keys():
                        reward = rVal - origResult[origId]
                    else:
                        # no result for original id: - -1
                        reward = rVal + 1
                    if self.reversed:
                        reward = 1.0- reward
                        if reward < -2.0:
                            reward = -2.0
                        elif reward > 2.0:
                            reward = 2.0
                    
                #print("reward: ", reward)
                cat = qId.split("-")[2]
                if not convId in self.rewardDict.keys():
                    self.rewardDict[convId] = dict()
                self.rewardDict[convId][cat] = reward


    def getPredictionProbabilityReward(self):
        
        origResult = dict()
        for conv in data_info:
            for turn in conv["questions"]:
                qId = turn['question_id']
                if not qId.endswith("-0-0"):
                    continue
                rVal = -1.0
                if len(turn["ranked_answers"])>0:
                    rVal = float(turn["ranked_answers"][0]["score"])    
                origResult[qId] = rVal
    
        for conv in data_info:
            for turn in conv["questions"]:
                rVal = -1.0
                if len(turn["ranked_answers"])>0:
                    rVal = float(turn["ranked_answers"][0]["score"])    

                qId = turn['question_id']
                convId = qId.split("-")[0] + "-" + qId.split("-")[1]
                if self.absolute:
                    if self.reversed:
                        reward = 1.0 -rVal
                    else:
                        reward = rVal
                else:
                    origId = convId + "-0-0"
                    if origId in origResult.keys():
                        reward = rVal - origResult[origId]
                    else:
                        # no result for original id: - -1
                        reward = rVal + 1
                    if self.reversed:
                        reward = 1.0- reward
                        if reward < -2.0:
                            reward = -2.0
                        elif reward > 2.0:
                            reward = 2.0
                    
                #print("reward: ", reward)
                cat = qId.split("-")[2]
                if not convId in self.rewardDict.keys():
                    self.rewardDict[convId] = dict()
                self.rewardDict[convId][cat] = reward


        return


    def storeRewardInfo(self):
        with open(self.outputFile, "w") as outFile:
            json.dump(self.rewardDict, outFile)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--inputfile", dest="inputfile")
    parser.add_argument("--rewardtype", dest="rewardtype")
    parser.add_argument("--outputfile", dest="outputfile")
    parser.add_argument("--reversed", action='store_true')
    parser.add_argument("--absolute", action='store_true')
    args = parser.parse_args()


    with open(args.inputfile, "r") as infile:
        data_info = json.load(infile)
  
    RewCol = RewardCollector(data_info, args.rewardtype, args.outputfile, args.absolute, args.reversed)