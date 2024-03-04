import json
import pickle

import spacy
import requests
import sys
sys.path.append("utils")
import utils as ut
import re
import regex
from argparse import ArgumentParser
from CLOCQInterfaceClient import CLOCQInterfaceClient


nlp = spacy.load("en_core_web_sm")
clocq = CLOCQInterfaceClient(host="https://clocq.mpi-inf.mpg.de/api", port="443")


def insertEntity(conversations):
  
    for entry in conversations:
        for q_entry in entry["questions"]:
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            question = q_entry["question"].replace("?", "")
            for ent in q_entry["conv_entities"]:
                #no mention in question detected, but entities available from history
                if ent["mention"] == "":
                    q_entry["reformulations"].append({"reformulation": question + " " + ent["label"] + "?",  "ref_type": "INS_ENT"})
            
    return conversations

def deleteEntity(conversations):
  
    for entry in conversations:
    
        for q_entry in entry["questions"]:
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            question = q_entry["question"]
            for ent in q_entry["conv_entities"]: 
                if not ent["mention"] == "":    
                    newQuest = regex.sub("(the )?"  +r'\b' +ent["mention"] +r'\b', "",  question, flags=re.IGNORECASE)
                    q_entry["reformulations"].append({"reformulation": newQuest,  "ref_type": "DEL_ENT"})
            
    return conversations

def insertPredicate(conversations):
    predAdds = []
  
    for entry in conversations:
        prevPredId = dict()
        for q_entry in entry["questions"]:
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
            question = q_entry["question"]
            qId = q_entry["question_id"]
            
            if int(qId[-1]) == 0:
                if "id" in q_entry["predicate"].keys():
                    prevPredId[0] = q_entry["predicate"]["id"]
                else:
                    prevPredId[0] = ""
                prevQuest = question
                continue
        
            predAdds = []
            if "id" in q_entry["predicate"].keys():
                currentPredId = q_entry["predicate"]["id"]
                key = int(qId[-1])-1
                if key in prevPredId.keys():
                    if currentPredId == prevPredId[key]:
                        if not "question_predicate_mention" in q_entry["predicate"].keys() or len(q_entry["predicate"]["question_predicate_mention"]) == 0:
                            print("same Predicate: ", question, "prev: ", prevQuest)
                            newQuest = question + " " + q_entry["predicate"]["label"]
                            predAdds.append({"reformulation": newQuest, "ref_type": "INS_PRED"})
                
           

            if "id" in q_entry["predicate"].keys():
                prevPredId[int(qId[-1])] = q_entry["predicate"]["id"]
            else:
                prevPredId[int(qId[-1])] = ""
            prevQuest = question

        if not "reformulations" in q_entry.keys():
            q_entry["reformulations"] = []
        
        if len(predAdds)> 0:
            q_entry["reformulations"].extend(predAdds)
        print("pred additions: ", predAdds)
        # print("predicate mention: ", predQuestion)
        print("----------------------------")     
    return conversations



def deletePredicate(conversations):
    predOmissions = []
    for entry in conversations:
        prevPredId = dict()
        for q_entry in entry["questions"]:
            question = q_entry["question"]
            qId = q_entry["question_id"]
            if int(qId[-1]) == 0:
                if "id" in q_entry["predicate"].keys():
                    prevPredId[0] = q_entry["predicate"]["id"]
                   
                else:
                    prevPredId[0] = ""
                prevQuest = question
                continue
           
            predOmissions = []
            if "id" in q_entry["predicate"].keys():
                currentPredId = q_entry["predicate"]["id"]
                key = int(qId[-1])-1
                if key in prevPredId.keys():
                    if currentPredId == prevPredId[key]:
                        if "question_predicate_mention" in q_entry["predicate"].keys() and len(q_entry["predicate"]["question_predicate_mention"]) > 0:
                            print("same Predicate: ", question, "prev: ", prevQuest)
                            newTypeQuest = question
                            for ment in q_entry["predicate"]["question_predicate_mention"]:
                                newTypeQuest = newTypeQuest.replace(ment, "")
                            if len(newTypeQuest)>2:
                        #  newTypeQuest = regex.sub(r'\b' + " ".join(q_entry["predicate"]["question_predicate_mention"]) +r'\b', "",  question, flags=re.IGNORECASE)
                                predOmissions.append({"reformulation": newTypeQuest, "ref_type": "DEL_PRED"})
                    
            if "id" in q_entry["predicate"].keys():
                prevPredId[int(qId[-1])] = q_entry["predicate"]["id"]
            else:
                prevPredId[int(qId[-1])] = ""
            prevQuest = question

        if not "reformulations" in q_entry.keys():
            q_entry["reformulations"] = []
        
        if len(predOmissions)> 0:
            q_entry["reformulations"].extend(predOmissions)
            print("pred omission: ", predOmissions)
        # print("predicate mention: ", predQuestion)
        print("----------------------------")     
                    
    return conversations 

def insertEntityTypes(conversations):
    for entry in conversations:
        for q_entry in entry["questions"]:
            question = q_entry["question"]
            entityTypes = []
           
            for ent in q_entry["entities"]:
                newTypeQuest = question
                #entity type only added if not already present
                if "type_mention" in ent.keys() :
                    continue
        
                currentQuest = newTypeQuest
                if ent["mention"] == "":
                    for etype in ent["type"]:
                        newTypeQuest = currentQuest.replace("?", "")
                        newTypeQuest = newTypeQuest + " "  + etype["label"] + "?"
                        #newTypeQuest = re.sub(r'(\w+) \1', r'\1', newTypeQuest, flags=re.IGNORECASE)
                        print("only type present", newTypeQuest)
                        entityTypes.append({"reformulation": newTypeQuest, "ref_type": "INS_ENT_TYPE"})
                else:
                    for etype in ent["type"]:
                        newTypeQuest = currentQuest
                        newTypeQuest = regex.sub("(the )?"  +r'\b' + ent["mention"] +r'\b', "the " + etype["label"] + " " + ent["mention"], newTypeQuest, flags=re.IGNORECASE)
                        #newTypeQuest = re.sub(r'(\w+) \1', r'\1', newTypeQuest, flags=re.IGNORECASE)
                        print("type + entity present: ", newTypeQuest)
                        entityTypes.append({"reformulation": newTypeQuest, "ref_type": "INS_ENT_TYPE"})
            
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
          
            q_entry["reformulations"].extend(entityTypes)
            print("entity types: ", entityTypes)
           # print("predicate mention: ", predQuestion)
            print("----------------------------")
            
    return conversations

def deleteEntityTypes(conversations):
    for entry in conversations:
        domain = entry["domain"]
        for q_entry in entry["questions"]:
            question = q_entry["question"]
            entityTypes = []
           
            for ent in q_entry["entities"]:
                
                #remove entity type if present
                if "type_mention" in ent.keys() :
                    newTypeQuest = regex.sub("(the )?"  +r'\b' + ent["type_mention"] +r'\b', "", question, flags=re.IGNORECASE)
                    q_entry["reformulations"].append({"reformulation": newTypeQuest, "ref_type": "DEL_ENT_TYPE"})
            
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
            q_entry["reformulations"].extend(entityTypes)
            print("entity types: ", entityTypes)
           # print("predicate mention: ", predQuestion)
            print("----------------------------")
            
    return conversations


def insertAnswerTypes(conversations):
    for entry in conversations:
        for q_entry in entry["questions"]:
            question = q_entry["question"]
            ansTypePhrases = []
            ansTypes = []
            #collect all answer types
            ansTypes =  q_entry["ans_types"] 
            print("ansTypes: ", ansTypes)
            #check if ans type already present and get question word 
            questWord =  q_entry["question_word"] 
            type_mention =   q_entry["ans_type_mention"]
            #we will only add type if not type is there already
            if not type_mention == "":
                continue

            humanType = False
            for aType in ansTypes:
                if aType == "human":
                    humanType = True
                    break
            
            if not questWord == "":
                splitQuestion = question.split(questWord)[1]
            else:
                splitQuestion = question
            
            doc = nlp(question)
            #verbIdx = 10000
            hasVerb = False
            for i in range(len(doc)):
                #print("verbidx: ", verbIdx)
                if doc[i].tag_.startswith("V"):
                    hasVerb = True
                    break
            if not hasVerb:
                splitQuestion = "is " + splitQuestion
            #add a few manual types
            if humanType:
                newQuest  = "Which person " + splitQuestion
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
                newQuest  = "Which individual " +  splitQuestion
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
            elif questWord.lower() == "where":
                newQuest  = "At which location " + splitQuestion
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
                newQuest  = "At which place " + splitQuestion
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
            elif questWord.lower() == "when":
                newQuest  = "At which point in time " + splitQuestion
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
            elif questWord.lower() == "how many":
                newQuest  = "which amount of " +  splitQuestion
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
                newQuest  = "which number of " + splitQuestion
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
            
            #add available answer types
            for aType in ansTypes:
                if aType == "string" or aType == "number" or aType == "existential":
                    continue
                if humanType or questWord.lower() == "who":
                    newQuest = " ".join(["Which", aType, splitQuestion])
                elif questWord.lower() == "where":
                    newQuest = " ".join(["At which", aType, splitQuestion])
                elif questWord.lower() == "when":
                    newQuest = " ".join(["At which", aType, splitQuestion])
                elif questWord == "":
                       newQuest = " ".join(["Which", aType, splitQuestion])
                else:
                    newQuest = " ".join([questWord, aType, splitQuestion])
                ansTypePhrases.append({"reformulation": newQuest, "ref_type": "INS_ANS_TYPE"})
           

            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
            q_entry["reformulations"].extend(ansTypePhrases)
            print("ans type phrases: ", ansTypePhrases)
            print("-------------------------")
    return conversations

def deleteAnswerTypes(conversations):
    for entry in conversations:
        for q_entry in entry["questions"]:
            question = q_entry["question"]
            ansTypes = q_entry["ans_types"] 
            type_mention = q_entry["ans_type_mention"]
            if not type_mention == "":
                noTypeQuest = regex.sub(  r'\b' + type_mention +r'\b', "", question, flags=re.IGNORECASE)
                q_entry["reformulations"].append({"reformulation": noTypeQuest, "ref_type": "DEL_ANS_TYPE"})
                        
    return conversations

def substitutePredicates(conversations):
    for entry in conversations:
    
        domain = entry["domain"]
        for q_entry in entry["questions"]:
           
            question = q_entry["question"]
            predParaphrases = []
            newPredPara = question
            q_entry["reformulations"].append({"reformulation": q_entry["paraphrase"], "ref_type": "SUBS_PRED"})
         
            #check if we have a predicate mention annotation
            if "id" in q_entry["predicate"].keys() and "question_predicate_mention" in q_entry["predicate"].keys():
                if len(q_entry["predicate"]["question_predicate_mention"]) == 0:
                    continue
                if not "manual_aliases" in q_entry["predicate"].keys():
                    continue

                pred_mention =  q_entry["predicate"]["question_predicate_mention"]
                print("predicate mention: ",pred_mention)
            
                for alias in q_entry["predicate"]["manual_aliases"]:
                    print("current alias: ", alias)
                    newPredPara = question
                    aliasdoc = nlp(alias)
                    aliasVerb = ""
                    #find out if alias includes a verb
                    for token in aliasdoc:
                        if token.tag_.startswith("V"):
                            aliasVerb += token.text + " "  
                    print("verb in alias: ", aliasVerb)
                    #both nouns: we can directly replace one with the other
                    
                    if aliasVerb == "" and not " ".join(pred_mention) == question:
                        alias = "is " + alias

                    for i in range(1,len(q_entry["predicate"]["question_predicate_mention"])):
                        #if notpred_mention[i] in alias:
                        newPredPara = re.sub("(the )?" +r'\b' +pred_mention[i] +r'\b', "" , newPredPara, flags=re.IGNORECASE)
                    newPredPara = re.sub("(the )?" + r'\b' +pred_mention[0] +r'\b', alias, newPredPara, flags=re.IGNORECASE)
                    
                    newPredPara = " ".join(newPredPara.split())
                    print("new pred para; ", newPredPara)
                    predParaphrases.append({"reformulation": newPredPara, "ref_type": "SUBS_PRED"})
                
                if "domain_aliases" in q_entry["predicate"].keys():
                    if domain in q_entry["predicate"]["domain_aliases"].keys():

                        for alias in q_entry["predicate"]["domain_aliases"][domain]:
                            print("current alias: ", alias)
                            newPredPara = question
                            aliasdoc = nlp(alias)
                            aliasVerb = ""
                            #find out if alias includes a verb
                            for token in aliasdoc:
                                if token.tag_.startswith("V"):
                                    aliasVerb += token.text + " "  
                            print("verb in alias: ", aliasVerb)
                            #both nouns: we can directly replace one with the other
                            
                            if aliasVerb == "" and not " ".join(pred_mention) == question:
                                alias = "is " + alias

                            for i in range(1,len(q_entry["predicate"]["question_predicate_mention"])):
                                #if notpred_mention[i] in alias:
                                newPredPara = re.sub("(the )?" +r'\b' +pred_mention[i] +r'\b', "" , newPredPara, flags=re.IGNORECASE)
                            newPredPara = re.sub("(the )?" + r'\b' +pred_mention[0] +r'\b', alias, newPredPara, flags=re.IGNORECASE)
                            
                            newPredPara = " ".join(newPredPara.split())
                            print("new pred para; ", newPredPara)
                            predParaphrases.append({"reformulation": newPredPara, "ref_type": "SUBS_PRED"})
               
                if "gender_aliases" in q_entry["predicate"].keys():
                    gender = ""
                    for ent in q_entry["entities"]:
                        genderId = clocq.connect(ent["id"], "P21") #"Q6581097"
                        
                        if not genderId is None:
                            if len(genderId)==1:
                                if len(genderId[0])>2:
                                    genderId = genderId[0][2]
                                else:
                                    genderId = ""
                            else:
                                genderId = ""
                       

                        if genderId == "Q6581097":
                            gender = "male"
                            break
                        elif genderId == "Q6581072":
                            gender = "female"
                            break

                    if gender in q_entry["predicate"]["gender_aliases"].keys():
                        print("gender: ", gender)
                        print("pred: ",  q_entry["predicate"])
                        for alias in q_entry["predicate"]["gender_aliases"][gender]:
                            print("current alias: ", alias)
                            newPredPara = question
                            aliasdoc = nlp(alias)
                            aliasVerb = ""
                            #find out if alias includes a verb
                            for token in aliasdoc:
                                if token.tag_.startswith("V"):
                                    aliasVerb += token.text + " "  
                            print("verb in alias: ", aliasVerb)
                            #both nouns: we can directly replace one with the other
                            
                            if aliasVerb == "" and not " ".join(pred_mention) == question:
                                alias = "is " + alias

                            for i in range(1,len(q_entry["predicate"]["question_predicate_mention"])):
                                #if notpred_mention[i] in alias:
                                newPredPara = re.sub("(the )?" +r'\b' +pred_mention[i] +r'\b', "" , newPredPara, flags=re.IGNORECASE)
                            newPredPara = re.sub("(the )?" + r'\b' +pred_mention[0] +r'\b', alias, newPredPara, flags=re.IGNORECASE)
                            
                            newPredPara = " ".join(newPredPara.split())
                            print("new pred para; ", newPredPara)
                            predParaphrases.append({"reformulation": newPredPara, "ref_type": "SUBS_PRED"})
                        
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
          
            q_entry["reformulations"].extend(predParaphrases)
            print("paraphrases: ", predParaphrases)
           
            print("----------------------------")            


    
    return conversations


def substituteEntity(conversations):
    for entry in conversations:
        for q_entry in entry["questions"]:
            question = q_entry["question"]

            entityParaphrases = []
           
            for ent in q_entry["entities"]:
                
                print("entity: ", ent)
                if ent["mention"] == "":
                    continue
                for alias in ent["aliases"]:
                    if alias.lower() == ent["mention"].lower():
                        continue
                    print("ent label ", ent["label"], " mention: ", ent["mention"], "alias: ", alias)
                    newAliasQuest = regex.sub(ent["mention"],alias, question, flags=re.IGNORECASE)
                    entityParaphrases.append({"reformulation":newAliasQuest, "ref_type": "SUBS_ENT"})
            
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
            
            q_entry["reformulations"].extend(entityParaphrases)
            print("entity paraphrase: ", entityParaphrases)
          
            print("----------------------------")
    return conversations



def substituteEntityTypes(conversations):
    for entry in conversations:
        for q_entry in entry["questions"]:
            question = q_entry["question"]
            entityTypes = []
    
            for ent in q_entry["entities"]:
                newTypeQuest = question
               #if type is present -> substitute it
                if "type_mention" in ent.keys():
                    for etype in ent["type"]:
                        if etype["label"] == ent["type_mention"]:
                            continue
                        newTypeQuest = question
                        newTypeQuest = regex.sub(r'\b' + ent["type_mention"] +r'\b',  etype["label"] , newTypeQuest, flags=re.IGNORECASE)
                     
                        entityTypes.append({"reformulation": newTypeQuest, "ref_type": "SUBS_ENT_TYPE"})
            
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
          
            q_entry["reformulations"].extend(entityTypes)
            print("entity types: ", entityTypes)
           # print("predicate mention: ", predQuestion)
            print("----------------------------")
            
    return conversations

def substituteAnswerTypes(conversations):
    for entry in conversations:
      
        for q_entry in entry["questions"]:
        
            question = q_entry["question"]
           
            ansTypePhrases = []
            ansTypes = q_entry["ans_types"] 
            type_mention = q_entry["ans_type_mention"]
            if type_mention == "":
                continue
            for ansType in ansTypes:
                if ansType.lower() == type_mention.lower():
                    continue
                newTypeQuest = regex.sub(r'\b' + type_mention +r'\b', ansType , question, flags=re.IGNORECASE)
                ansTypePhrases.append({"reformulation": newTypeQuest, "ref_type": "SUBS_ANS_TYPE"})
    
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
          
            q_entry["reformulations"].extend(ansTypePhrases)
            print("ans types: ", ansTypePhrases)
           # print("predicate mention: ", predQuestion)
            print("----------------------------")
            
    return conversations

def replaceEntitiesByTypes(conversations):
    for entry in conversations:
    
        for q_entry in entry["questions"]:
            question = q_entry["question"]
            
            qId = q_entry["question_id"]
            if int(qId[-1]) == 0:
                continue

           
            entityTypes = []
            question = ' '.join(question.split())
            
            print(" question: ",q_entry["question"])


            for ent in q_entry["conv_entities"]:
                newTypeQuest = question
                if ent["mention"] == "":
                    print("no entity mention for ent: ", ent)
                    continue
                #type info already present in question: then only remove entity
                if "type_mention" in ent.keys():
                    newTypeQuest = regex.sub("(the )?"  +r'\b' +ent["mention"] +r'\b', "",  newTypeQuest, flags=re.IGNORECASE)
                    entityTypes.append({"reformulation": newTypeQuest, "ref_type": "REPLACE_ENT_BY_TYPE"})
                    continue
                if len(ent["type"]) == 0:
                    continue
                #otherwise replace entity with a type - TODO: should we use all types here?
                for etype in ent["type"]:
                    typeRep = etype["label"]
           
                    newTypeQuest = regex.sub("(the )?"  +r'\b' +ent["mention"] +r'\b', "the " + typeRep, newTypeQuest, flags=re.IGNORECASE)
                    #newTypeQuest = re.sub(r'(\w+) \1', r'\1', newTypeQuest, flags=re.IGNORECASE)
                    print("new type quest3; ", newTypeQuest)
                    entityTypes.append({"reformulation": newTypeQuest, "ref_type": "REPLACE_ENT_BY_TYPE"})
            
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
          
            q_entry["reformulations"].extend(entityTypes)
            print("entity types: ", entityTypes)
           # print("predicate mention: ", predQuestion)
            print("----------------------------")     
                       

    return conversations


def replaceEntitiesByPronouns(conversations):
    for entry in conversations:

        for q_entry in entry["questions"]:
            question = q_entry["question"]
            
            qId = q_entry["question_id"]
            if int(qId[-1]) == 0:
                continue
           
            entityTypes = []
            question = ' '.join(question.split())
       
            print(" question: ",q_entry["question"])


            for ent in q_entry["conv_entities"]:
                newTypeQuest = question
                if ent["mention"] == "":
                    print("no entity mention for ent: ", ent)
                    continue
                if "type_mention" in ent.keys():
                    newTypeQuest = regex.sub("(the )?"  +r'\b' +ent["type_mention"] +r'\b', "",  newTypeQuest, flags=re.IGNORECASE)
                   
             
                genderId =  clocq.connect(ent["id"], "P21")

                if not genderId is None:
                    if len(genderId)==1:
                        if len(genderId[0])>2:
                            genderId = genderId[0][2]
                        else:
                            genderId = ""
                    else:
                        genderId = ""
              
                if genderId == "Q6581097":
                    newTypeQuest = regex.sub("(the )?"  + ent["mention"], "he", newTypeQuest, flags=re.IGNORECASE)
                elif genderId == "Q6581072":
                    newTypeQuest = regex.sub("(the )?"  + ent["mention"], "she", newTypeQuest, flags=re.IGNORECASE)
                
                else:
                    
                    newTypeQuest = regex.sub("(the )?"  + ent["mention"], "it", newTypeQuest, flags=re.IGNORECASE)
            
                print("new type quest3; ", newTypeQuest)
                entityTypes.append({"reformulation": newTypeQuest, "ref_type": "REPLACE_ENT_BY_PRONOUN"})
            
            if not "reformulations" in q_entry.keys():
                q_entry["reformulations"] = []
            
          
            q_entry["reformulations"].extend(entityTypes)
            print("entity types: ", entityTypes)
           # print("predicate mention: ", predQuestion)
            print("----------------------------")     
                       
    return conversations


if __name__ == "__main__":
    parser = ArgumentParser()
 
    parser.add_argument("--inpath", dest="inpath")
    parser.add_argument("--outpath", dest="outpath")
    args = parser.parse_args()
   
    with open(args.inpath, "r", encoding='utf-8') as convFile:
        conversations = json.load(convFile)

    #add corresponding reformulations to dataset by applying each rule where applicable
    conversations = insertEntity(conversations)
    conversations = deleteEntity(conversations)
    conversations = insertEntityTypes(conversations)
    conversations = deleteEntityTypes(conversations)
    conversations = insertAnswerTypes(conversations)
    conversations = deleteAnswerTypes(conversations)
    conversations = substituteEntity(conversations)
    conversations = substitutePredicates(conversations)
    conversations = insertPredicate(conversations)
    conversations = deletePredicate(conversations)
    conversations = substituteEntityTypes(conversations)
    conversations = substituteAnswerTypes(conversations)
    conversations = replaceEntitiesByTypes(conversations)
    conversations = replaceEntitiesByPronouns(conversations)


    with open(args.outpath, "w", encoding='utf-8') as convFile:
        json.dump(conversations, convFile)

    