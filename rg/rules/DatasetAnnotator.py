import json
import pickle
import sys
sys.path.append("utils")
import utils as ut
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity as cs
from QAEncoding import QAEncoder
from sentence_transformers import SentenceTransformer, util
import re
import regex
from argparse import ArgumentParser
from CLOCQInterfaceClient import CLOCQInterfaceClient



class DatasetAnnotator():

    def __init__(self, conversations):
        self.sentenceTransformer = SentenceTransformer('all-mpnet-base-v2') #'all-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_sm")
        self.qaEnc = QAEncoder(tagmeThreshold=0.2)
        self.conversations = conversations
        self.stored_entities = dict()
        self.clocq = CLOCQInterfaceClient(host="https://clocq.mpi-inf.mpg.de/api", port="443")

        self.domain_types = {
            "books": ["book", "book series",  "novel series", "novel", "series", "writer", "author", "novelist", "character"],
            "tvseries": ["TV series", "television series", "series", "show", "actor", "actress", "character", "director"],
            "music":["music album", "band", "music band", "album", "song", "record label"],
            "movies": ["movie", "film", "actor", "actress", "character", "director"],
            "soccer":["soccer player", "player", "football club", "club"]
        }

    def annotate(self):
        for entry in conversations:
            questionData = self.getConversationalEntities(entry["questions"])
            domain = entry["domain"]
            for q_entry in questionData:
                self.annotateEntities(q_entry)
                self.annotateEntityTypes(q_entry, domain)
                self.annotateAnswerTypes(q_entry)
                self.annotatePredicates(q_entry, domain)
              
                


    def annotateEntities(self, q_entry):
        #ner to get some additional entities
        if "completed" in q_entry.keys():
            self.stored_entities[q_entry["question_id"]] =self.qaEnc.getQuestionEntities(q_entry["completed"])
        else:
            self.stored_entities[q_entry["question_id"]] =self.qaEnc.getQuestionEntities(q_entry["question"])
        #retrieve entity aliases
        for ent in q_entry["entities"]:
            result  = self.clocq.get_aliases(ent["id"])
            ent["aliases"] = []
            for alias in result:
                alias = alias.replace("\\", "")
                ent["aliases"].append(alias)
        #detect entity mention in actual question
        self.getEntityMention(q_entry)
        return 


    def annotateEntityTypes(self, q_entry, domain):
        #retrieve KG types for entities
        for ent in q_entry["entities"]:
            ent["type"] = self.clocq.get_types(ent["id"])
        #find mention of entity type in question
        self.getEntityTypeMention(q_entry, domain)
      

    def annotateAnswerTypes(self, q_entry):
        ansTypes = []
        #retrieve KG type information of answers
        for ans in q_entry["answers"]:
            if ans["id"].startswith("Q"):
                ans["type"] = self.clocq.get_types(ans["id"])
                for atype in ans["type"]:
                    if "label" in atype:
                        if not atype["label"] in ansTypes:
                            ansTypes.append(atype["label"])
                    else:
                        print("list but one is string: ")
                        if not atype in ansTypes:
                            ansTypes.append(atype)
            else:
                if ut.is_timestamp(ans["id"]):
                    if " " in ans["label"]:
                        ans["type"] = "date"
                    else:
                        ans["type"] = "year"
                elif ans["id"] == "Yes" or ans["id"] == "No":
                    ans["type"] = "existential"
                elif ut.hasNumber(ans["id"]):
                    ans["type"] = "number"
                else:
                    ans["type"] = "string"
                ansTypes.append(ans["type"])
        
        ansTypes = list(set(ansTypes))
        q_entry["ans_types"] = ansTypes
        #find mention of answer type if present in question
        self.getAnsTypeMention(q_entry, ansTypes)



    def annotatePredicates(self, q_entry, domain):
        if "completed" in q_entry.keys():
            completeQuestion = q_entry["completed"]
        else:
            completeQuestion = q_entry["question"]
        q_entry["predicate"] = []
        qId = q_entry["question_id"]
        ned_entities = self.stored_entities[qId]
        predicate = []
        fact_labels = []
        fact_ids = []
        for ent in q_entry["entities"]:
            #try to connect question and answer entities
            for ans in q_entry["answers"]:
                if ans["id"].startswith("Q"):
                    p_facts = self.clocq.connect(ent["id"], ans["id"])
                    if not p_facts is None:
                        if len(p_facts)> 1000:
                            p_facts = p_facts[:1000]
                        for fact in p_facts:
                            if isinstance(fact[0], list):
                                continue
                            predicate.append({"id": fact[1], "label": self.clocq.get_label(fact[1])})
                            labelString = ""
                            for f in fact:
                                labelString += self.clocq.get_label(f) + " "
                            fact_labels.append( labelString)
                            fact_ids.append(fact)

            if len(predicate) == 0:
                #if no connections available use neighborhood search for question entities and try to find answer in returned top facts
                p_facts = self.clocq.get_neighborhood(ent["id"])
                for fact in p_facts:
                    labelString = ""
                    found = False
                    for i in range(len(fact)):
                        for ans in q_entry["answers"]:
                            if ans["id"] == fact[i]["id"].strip("\""):
                                if fact[i-1]["id"].startswith("P"):
                                    #TODO: maybe check if it is better if take all preds and not just this one
                                    predicate.append(fact[i-1])
                                    #   print("before")
                                elif fact[i+1]["id"].startswith("P"):
                                    predicate.append(fact[i+1])
                                # print("neighborhood successful")
                                found = True
                                break
                    if found:
                        idList = []
                        for i in range(len(fact)):
                            labelString += fact[i]["label"] + " "
                            idList.append(fact[i]["id"])
                                
                        fact_labels.append(labelString)
                        fact_ids.append(idList)

        if len(predicate)== 0:  
            #not successful yet: try to find facts connecting question entities (especially when answer is count or yes/no)
            for ent in q_entry["entities"]:
                if not ent in ned_entities:
                    ned_entities.append(ent)
            # print("ned_ents after: ", ned_entities)
            for i in range(len(ned_entities)):
                for j in range(i+1,len(ned_entities)):
                    if "mention" in ned_entities[i].keys():
                        if "mention" in  ned_entities[j].keys():
                            if ned_entities[i]["mention"] == ned_entities[j]["mention"]:
                                continue
                    p_facts = self.clocq.connect(ned_entities[i]["id"], ned_entities[j]["id"])
                    
                    if not p_facts is None:
                        if len(p_facts)> 1000:
                            p_facts = p_facts[:1000]
                
                        for fact in p_facts:
                            if isinstance(fact[0], list):
                                continue
                            predicate.append({"id": fact[1], "label": self.clocq.get_label(fact[1])})
                            labelString = ""
                            for f in fact:
                                labelString += self.clocq.get_label(f) + " "
                            fact_labels.append(labelString)
                            fact_ids.append(fact)
            if len(predicate) == 0:
                for k in range(len(ned_entities)):
                    p_facts = self.clocq.get_neighborhood(ned_entities[k]["id"])
                    for fact in p_facts:
                        labelString = ""
                        found = False
                        for i in range(len(fact)):
                            for ans in q_entry["answers"]: # answers not useful here (count/yes/no)??
                                if ans["id"] == fact[i]["id"].strip("\""):
                                    if fact[i-1]["id"].startswith("P"):
                                        #TODO: maybe check if it is better if take all preds and not just this one
                                        predicate.append(fact[i-1])
                                        #print("before")
                                    elif fact[i+1]["id"].startswith("P"):
                                        #print("after")
                                        predicate.append(fact[i+1])
                                    #print("neighborhood successful")
                                    found = True
                                    break
                        if not found:
                            for i in range(len(fact)):
                                for j in range(0,len(ned_entities)):
                                    if "mention" in ned_entities[k].keys():
                                        if "mention" in  ned_entities[j].keys():
                                            if ned_entities[k]["mention"] == ned_entities[j]["mention"]:
                                                continue
                                            if ned_entities[j]["id"] == fact[i]["id"].strip("\""):
                                                if fact[i-1]["id"].startswith("P"):
                                                    #TODO: maybe check if it is better if take all preds and not just this one
                                                    predicate.append(fact[i-1])
                                                    #print("before")
                                                elif fact[i+1]["id"].startswith("P"):
                                                    # print("after")
                                                    predicate.append(fact[i+1])
                                                #print("neighborhood successful")
                                                found = True
                                                break
                        if found:
                            idList = []
                            for i in range(len(fact)):
                                labelString += fact[i]["label"] + " "
                                idList.append(fact[i]["id"])
                                
                            fact_labels.append(labelString)
                            fact_ids.append(idList)

            
            q_entry["predicate"] = dict()
            q_entry["answerfact"] = dict()  
        if len(predicate)>1:
            #if we have several candidates: get top fact with sentencebert sim between facts and completed question
            encoded_question = self.sentenceTransformer.encode(completeQuestion)
            #print("encoded question", encoded_question)
            encoded_facts= self.sentenceTransformer.encode(fact_labels) 
            sim = []
            max_sim = 0
            max_idx =0
            
            for i in range(len(encoded_facts)):
                try:
                    sim = util.cos_sim(encoded_question, np.expand_dims(encoded_facts[i], axis=0))#cs([encoded_question[0], encoded_facts[i]])
                    #print("sim: ", sim)
                    if float(sim[0][0]) > max_sim:
                        max_sim = float(sim[0][0])
                        max_idx = i
                        # print("maxval: ", max_idx, "fact: ", fact_labels[max_idx], "prd: ", predicate[max_idx], "maxsim: ", max_sim)
                except KeyError:
                    continue 
            if isinstance(predicate[max_idx], list):
                retPred = predicate[max_idx][0]
            else:
                retPred = predicate[max_idx]
            q_entry["predicate"] = retPred
            q_entry["answerfact"] = {"id": fact_ids[max_idx], "label": fact_labels[max_idx]}
        elif len(predicate)==1:
            if isinstance(predicate, list):
                retPred = predicate[0]
            else:
                retPred = predicate
            q_entry["predicate"] = retPred
            q_entry["answerfact"] =  {"id": fact_ids[0], "label": fact_labels[0]}

        
       
        self.extractPredicateAliases(q_entry)
        self.getPredicateMention(q_entry, domain)

        return


    def getConversationalEntities(self,questionData):
 
        for q_entry in questionData:
            convEntities = []
            convEntIds = []
          
            turn = int(q_entry["turn"])
            if turn == 0:
                q_entry["conv_entities"] = []
                continue
            for i in range(turn):
                for ent in q_entry["entities"]:
                    oldEnts = questionData[i]["entities"]
                    for oldEnt in oldEnts:
                        if ent["id"] == oldEnt["id"]:
                            if not ent["id"] in convEntIds:
                                convEntIds.append(ent["id"])
                                convEntities.append(ent)
                    oldAns = questionData[i]["answers"]
                    for oldA in oldAns:
                        if ent["id"] == oldA["id"]:
                            if not ent["id"] in convEntIds:
                                convEntIds.append(ent["id"])
                                convEntities.append(ent)
                  
            q_entry["conv_entities"] = convEntities
            print("convEntities: ", convEntities)
        return questionData
    

    def getEntityMention(self, q_entry):
   
        question = q_entry["question"]
        qId = q_entry["question_id"]
        tagged_entities = self.stored_entities[qId]
        for ent in q_entry["entities"]:
            ent["mention"] = ""
            lab_matched = re.search(r'\b' + re.escape(ent["label"]) +r'\b', question, flags=re.IGNORECASE | re.UNICODE)
            if lab_matched:
                ent["mention"] = ent["label"]
            else:
                found = False
                for alias in ent["aliases"]:
                    al_matched = re.search(r'\b' + re.escape(alias) +r'\b', question, flags=re.IGNORECASE | re.UNICODE)
                    if al_matched:
                        ent["mention"] = alias
                        found = True
                        break
                if not found:
                    for tagged_ent in tagged_entities:
                        if tagged_ent["id"] == ent["id"]:
                            print("ent id match: ", ent["id"])
                            if not "mention" in tagged_ent.keys():
                                tagged_ent["mention"] = tagged_ent["label"]
                            tag_matched = re.search(r'\b' + re.escape(tagged_ent["mention"]) +r'\b', question,  flags=re.IGNORECASE | re.UNICODE)
                            if tag_matched:
                                ent["mention"] = tagged_ent["mention"]
        print("entry: ", q_entry)
        return                     

    
    def getEntityTypeMention(self, q_entry, domain):
        #if we previously added entity then some LM-based sentence correction might be needed otherwise type check here not so helpful
            question = q_entry["question"]
        
            for ent in q_entry["entities"]:
                if ent["mention"] == "":
                    print("no entity mention for ent: ", ent)
                    continue
                entMatch =  re.search(ent["mention"], question)
                if not entMatch:
                    continue
            
                for dtype in self.domain_types[domain]:
                    pattern = "(the )?" + dtype
                    dtypeMatch = re.search(pattern, question, flags=re.IGNORECASE)
                    print("dtypeMatch:", dtypeMatch)
                    if not dtypeMatch:
                        continue
                    
                    if int(entMatch.span()[0]) == (dtypeMatch.span()[1] + 1) or int(entMatch.span()[1]) == (dtypeMatch.span()[0] - 1):
                        ent["type_mention"] = dtypeMatch.group(0)
                        break

                for etype in ent["type"]:   
                    if etype in self.domain_types[domain]:
                        continue
                    pattern = "(the )?" + etype["label"]
                    dtypeMatch = re.search(pattern, question, flags=re.IGNORECASE)
                    print("dtypeMatch:", dtypeMatch)
                    if dtypeMatch: 
                        if int(entMatch.span()[0]) == (dtypeMatch.span()[1] + 1) or int(entMatch.span()[1]) == (dtypeMatch.span()[0] - 1): 
                            ent["type_mention"] = dtypeMatch.group(0)

            for ent in q_entry["conv_entities"]:
                if ent["mention"] == "":
                    print("no entity mention for ent: ", ent)
                    continue
                entMatch =  re.search(ent["mention"], question)
                if not entMatch:
                    continue
            
                for dtype in self.domain_types[domain]:
                    pattern = "(the )?" + dtype
                    dtypeMatch = re.search(pattern, question, flags=re.IGNORECASE)
                    print("dtypeMatch:", dtypeMatch)
                    if not dtypeMatch:
                        continue
                    
                    if int(entMatch.span()[0]) == (dtypeMatch.span()[1] + 1) or int(entMatch.span()[1]) == (dtypeMatch.span()[0] - 1):
                        ent["type_mention"] = dtypeMatch.group(0)
                        break

                for etype in ent["type"]:   
                    if etype in self.domain_types[domain]:
                        continue
                    pattern = "(the )?" + etype["label"]
                    dtypeMatch = re.search(pattern, question, flags=re.IGNORECASE)
                    print("dtypeMatch:", dtypeMatch)
                    if dtypeMatch: 
                        if int(entMatch.span()[0]) == (dtypeMatch.span()[1] + 1) or int(entMatch.span()[1]) == (dtypeMatch.span()[0] - 1): 
                            ent["type_mention"] = dtypeMatch.group(0)

            return 


    def getAnsTypeMention(self, q_entry, ansTypes):
        question = q_entry["question"]
        questWord = ""
        type_mention = ""
        print("ansTypes: ", ansTypes)
        for wh_word in ["whom", "who", "what", "which", "how many", "when", "where", "how"]:
            whMatch = re.search(wh_word, question, flags=re.IGNORECASE) 
            if not whMatch is None:
                questWord = whMatch.group(0)
                break
        
        if not questWord == "":
            splitQuestion = question.split(questWord)[1]
            for ansType in ansTypes:
                if splitQuestion.startswith(ansType):
                    type_mention= ansType
                    q_entry["question_word"] = questWord
                    q_entry["ans_type_mention"] = type_mention
                    return 
        q_entry["question_word"] = questWord

        doc = self.nlp(question)
        for i in range(len(doc)):
            if doc[i].tag_.startswith("W"):
                if doc[i+1].pos_ == "NOUN":
                    if i+2 < len(doc) and doc[i+2].tag_.startswith("V"):
                        #in this case the noun is very likely a type:
                        type_mention = doc[i+1].text
                    elif i+3 < len(doc) and doc[i+3].tag_.startswith("V"):
                        type_mention = doc[i+1].text + " " + doc[i+2].text
        q_entry["ans_type_mention"] = type_mention
        return 



    def extractPredicateAliases(self, q_entry):
        if "id" in q_entry["predicate"].keys():
            results = self.qaEnc.getAliasesSparql(q_entry["predicate"]["id"])
            for result in results["results"]["bindings"]:
                if "wdAltLabel" in result.keys():
                    q_entry["predicate"]["aliases"] = result["wdAltLabel"]["value"].split(",")
        return 


    def getPredicateMention(self, q_entry, domain):
        mentionMatch  = 0
        predCount = 0
        predQuestion = q_entry["question"]
        keyString = "question_predicate_mention"

        predicate = q_entry["predicate"]
        print("predicate: ", predicate)
        
        found = False
        questdoc = self.nlp(predQuestion)
        lemmaQuest = ""
        verbQuest = []
        #lemmatize question
        for token in questdoc:
            #if token.is_stop:
                #   continue
            lemmaQuest += token.lemma_ + " "
            if token.tag_.startswith("V"):
                verbQuest.append(token.text)
        print("lemmaQuest: ", lemmaQuest)
        print("verbquest: ", verbQuest)
        predicate[keyString] = ""
        if "label" in predicate.keys():
            predCount += 1
            labeldoc = self.nlp(predicate["label"])
            lemmaLabel = ""
            lemmaText = ""
            #lemmatize predicate label
            for token in labeldoc:
                #if token.is_stop:
                    #   continue
                lemmaLabel += token.lemma_ + " "
                lemmaText += token.text + " "
            lemmaLabel = "".join(lemmaLabel.rstrip())
            lemmaText = "".join(lemmaText.rstrip())
            if lemmaLabel in lemmaQuest:
                verbQuest.append(lemmaText)
                predicate[keyString] = verbQuest#predicate["label"]
                print("label match", predicate[keyString])
                found = True
                mentionMatch += 1
            print("lemmaLabel: ", lemmaLabel)
        
        if not found:
            if "aliases" in predicate.keys(): 
            #lemmaAliasList = []
                for i in range(len(predicate["aliases"])):
                    aliasedoc = self.nlp(predicate["aliases"][i])
                    lemmaAlias = ""
                    lemmaText = ""
                    #lemmatize predicate aliases
                    for token in aliasedoc:
                        #   if token.is_stop:
                        #      continue
                        lemmaAlias += token.lemma_ + " "
                        lemmaText += token.text + " "
                    lemmaAlias = "".join(lemmaAlias.rstrip())
                    lemmaText = "".join(lemmaText.rstrip())
                    print("lemma alias: ", lemmaAlias)
                    if lemmaAlias in lemmaQuest:
                        verbQuest.append(lemmaText)
                        predicate[keyString] = verbQuest #predicate["aliases"][i]
                        print("alias match", predicate["aliases"][i])
                        found = True
                        mentionMatch += 1
                        break
        if not found:
            #we remove entities, type info and question words and treat what is left as predicate mention
            for ent in q_entry["entities"]:
                predQuestion =  regex.sub("(the )?"  +r'\b'+ ent["label"] +r'\b', "", predQuestion, flags=re.IGNORECASE)
                for etype in ent["type"]:
                #if etype["label"].lower() in completeQuest.lower():
                    print("pred before type: ", predQuestion)
                    predQuestion  = regex.sub("(the )?"  +r'\b'+  etype["label"] +r'\b', "", predQuestion, flags=re.IGNORECASE)
                    print("pred after type: ", predQuestion)
            qId = q_entry["question_id"]
            tagged_entities = self.stored_entities[qId]
            #also remove entities found by tagme (could be more than the annotated ones, sometimes also types)
            for tagged_ent in tagged_entities:
                if not "mention" in tagged_ent.keys():
                    print("NO MENTION for ", tagged_ent)
                else:
                    print("pred before mention: ", predQuestion)
                    predQuestion =  regex.sub("(the )?" + r'\b' + tagged_ent["mention"]  +r'\b', "", predQuestion, flags=re.IGNORECASE)
                    print("pred after mention: ", predQuestion)
                        

            for wh_word in ["whom", "who", "what", "which", "how many", "when", "where", "how"]:
                #if wh_word.lower() in predQuestion.lower():
                predQuestion =  regex.sub("(in|from|at|after )?" + wh_word, "", predQuestion, flags=re.IGNORECASE)

            for dom_type in self.domain_types[domain]:
                predQuestion =  regex.sub("(the )?" + dom_type, "", predQuestion, flags=re.IGNORECASE)
            
            #for time_word in TIME_WORDS:
            predQuestion =  regex.sub("(in|from|after)? [12][0-9]{3}" , "", predQuestion, flags=re.IGNORECASE)

            doc = self.nlp(predQuestion)
            print("pred quest: ", predQuestion)
            predQuestion = ""
            #further remove stop words and proper nouns
            for k in range(len(doc)):
                if doc[k].is_stop and not  doc[k].tag_.startswith("V"):
                    continue
                if  doc[k].pos_ == "PROPN":
                    continue
                if  doc[k].text in ["?", "/", '"' ":", ",", "-", "(", ")", "."]:
                    continue
                if k == len(doc)-1:
                    predQuestion +=  doc[k].text
                else:
                    predQuestion +=  doc[k].text + " "
                    
            predQuestion = " ".join(predQuestion.split())  
            predicate[keyString] = predQuestion.split(" ")
            
            for verb in verbQuest:
                
                if not verb in predicate[keyString]:
                    add = True
                    #make sure that verb is not already contained in longer mention
                    for i in range(len(predicate[keyString])):
                        if verb in predicate[keyString][i]:
                            add = False
                            break
                    if add:
                        predicate[keyString].append(verb)
            if "" in predicate[keyString]:
                predicate[keyString].remove("")  
            elif " " in predicate[keyString]:
                predicate[keyString].remove(" ")

            
        print("pred mention: ", predicate[keyString])
        print("-------------------------------")
        

        return 
    
    def postProcessPredicates(self, conversations, annotatedPreds):
        freqPreds = self.getMostFrequentPredicates(conversations)
        for entry in conversations:
            domain = entry["domain"]
            for q_entry in entry["questions"]:
                self.detectPredicatesFromList(freqPreds, q_entry)
                if annotatedPreds != "":
                    self.addManualAnnotations(annotatedPreds, q_entry)
                self.extractPredicateAliases(q_entry)
                self.getPredicateMention(q_entry, domain)


    def addManualAnnotations(self, pred_list, q_entry):

        if not "id" in q_entry["predicate"].keys():
            return
        
        for pred in pred_list:
            if pred["id"] == q_entry["predicate"]["id"]:
                if "manual_aliases" in pred:
                    q_entry["predicate"]["manual_aliases"] = pred["manual_aliases"]
                if "domain_aliases" in pred:
                    q_entry["predicate"]["domain_aliases"] = pred["domain_aliases"]
                if "gender_aliases" in pred:
                    q_entry["predicate"]["gender_aliases"] = pred["gender_aliases"]
        return 


    def getMostFrequentPredicates(self, conversations):
        predGroupsAll = []
        for entry in conversations:
            for q_entry in entry["questions"]:
                if not "id" in q_entry["predicate"].keys():
                    continue
                pId = q_entry["predicate"]["id"]
                if int(q_entry["turn"]) == 0:
                    completeQuest = q_entry["question"]
                else:
                    completeQuest = q_entry["completed"]
                found = False
                
                for group in predGroupsAll:
                    if group["predicate"]["id"] == pId:
                        if "mention" in q_entry["predicate"].keys():
                            group["predicate"]["mention"].append(q_entry["predicate"]["mention"])

                        group["examples"].append({"question_id":q_entry["question_id"], "completed": completeQuest})
                        found = True
                if not found:
                    predGroups = dict()
                # predGroups["id"] = pId
                    predGroups["predicate"] = q_entry["predicate"]
                    if "mention" in q_entry["predicate"].keys():
                        predGroups["predicate"]["mention"] = q_entry["predicate"]["mention"]
                    else:
                        predGroups["alias_mentions"] = []
                    predGroups["examples"] = [{"question_id":q_entry["question_id"], "completed": completeQuest}]
                    predGroupsAll.append(predGroups)

        relevantPreds = []
        for group in predGroupsAll:
            if len(group["examples"])> 3:
                relevantPreds.append(group["predicate"])
        print("number of predicates: ", len(predGroupsAll))
        print("relevant groups: ", relevantPreds)
        print("relevant groups len: ", len(relevantPreds))

        with open("frequent_predicates.json", "w") as convFile:
            json.dump(relevantPreds, convFile)

        return relevantPreds
    

    def detectPredicatesFromList(self, pred_list, q_entry):
        if "completed" in q_entry.keys():
            completeQuestion = q_entry["completed"]
        else:
            completeQuestion = q_entry["question"]
        if isinstance(q_entry["predicate"], list):
            if len(q_entry["predicate"]) == 0:

                for pred in pred_list:
                    if pred["label"].lower() in completeQuestion.lower():
                        q_entry["predicate"] = [pred]
                        break
                  
        return 


if __name__ == "__main__":
    parser = ArgumentParser()
 
    parser.add_argument("--inpath", dest="inpath")
    parser.add_argument("--outpath", dest="outpath")
    parser.add_argument("--annotatedPreds", dest="annotatedPreds", default="")
    args = parser.parse_args()
   
    with open(args.inpath, "r", encoding='utf-8') as convFile:
        conversations = json.load(convFile)

    annotatedPreds = ""
    if args.annotatedPreds != "":
        with open(args.annotatedPreds, "r", encoding='utf-8') as convFile:
            annotatedPreds = json.load(convFile)

    conversations = conversations[:10]
    annotator = DatasetAnnotator(conversations)
    annotator.annotate()
    annotator.postProcessPredicates(conversations, annotatedPreds)

    with open(args.outpath, "w", encoding='utf-8') as convFile:
        json.dump(conversations, convFile)
