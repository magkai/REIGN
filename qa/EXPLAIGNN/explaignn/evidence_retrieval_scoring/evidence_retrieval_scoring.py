import json
import os
from pathlib import Path
from tqdm import tqdm

from explaignn.evaluation import answer_presence
from explaignn.library.utils import get_logger


class EvidenceRetrievalScoring:
    """Abstract class for ERs phase."""

    def __init__(self, config):
        """Initialize ERS module."""
        self.config = config
        self.logger = get_logger(__name__, config)

    def train(self, *args):
        """Method used in case no training required for ERS phase."""
        self.logger.info("Module used does not require training.")

    def inference(self, sources=None):
        """Run ERS on data and add retrieve top-e evidences for each source combination."""
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        qu = self.config["qu"]
        ers = self.config["ers"]
        method_name = self.config["name"]
        

        # either use given option, or from config
        if not sources is None:
            source_combinations = [sources]
        else:
            source_combinations = self.config["source_combinations"]

        # go through all combinations
        for sources in source_combinations:
            sources_string = "_".join(sources)

            input_path = os.path.join(input_dir, qu, f"train_qu-{method_name}.json")
            output_path = os.path.join(
                output_dir, qu, ers, sources_string, f"train_ers-{method_name}.jsonl"
            )
            self.inference_on_data_split(input_path, output_path, sources, train=True)

            if self.config.get("ers_update_clocq_cache", True):
                self.store_cache()

            input_path = os.path.join(input_dir, qu, f"dev_qu-{method_name}.json")
            output_path = os.path.join(
                output_dir, qu, ers, sources_string, f"dev_ers-{method_name}.jsonl"
            )
            self.inference_on_data_split(input_path, output_path, sources)

            if self.config.get("ers_update_clocq_cache", True):
                self.store_cache()

            input_path = os.path.join(input_dir, qu, f"test_qu-{method_name}.json")
            output_path = os.path.join(
                output_dir, qu, ers, sources_string, f"test_ers-{method_name}.jsonl"
            )
            self.inference_on_data_split(input_path, output_path, sources)

        # store results in cache (if applicable)
        if self.config.get("ers_update_clocq_cache", True):
            self.store_cache()

    def inference_on_data_split(self, input_path, output_path, sources, train=False):
        """
        Run ERS on the dataset to predict
        answering evidences for each SR in the dataset.
        """
        # open data
        with open(input_path, "r") as fp:
            data = json.load(fp)
        self.logger.info(f"Input data loaded from: {input_path}.")

        # score
        answer_presences = list()
        source_to_ans_pres = {"kb": 0, "text": 0, "table": 0, "info": 0, "all": 0}

        # create folder if not exists
        output_dir = os.path.dirname(output_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        answerable_convs = []
        max_pos_evidences = self.config["gnn_train_max_pos_evidences"] if train else 0
        print("max pos evidences: ", max_pos_evidences)
        # process data
        with open(output_path, "w") as fp:
            answerable_path = output_path.replace(".jsonl", "_answerable.jsonl")
            aCount = 0
            with open(answerable_path, "w") as fp2:
                for conversation in tqdm(data):
                    newConv = dict()
                    if "conversation_id" in conversation.keys():
                        newConv["conversation_id"] = conversation["conversation_id"]
                    newConv["questions"] = []
                    for turn in conversation["questions"]:
                        top_evidences = self.inference_on_turn(turn, sources)
                        if top_evidences is None:
                            turn["top_evidences"] = []
                            turn["answer_presence"] = False
                            turn["answer_presence_per_src"] = {"kb":0}
                            continue
                        
                        turn["top_evidences"] = top_evidences

                        # answer presence
                        hit, answering_evidences = answer_presence(top_evidences, turn["answers"])
                        turn["answer_presence"] = hit
                        # prune instances with too many answering evidences (likely to be spurious path)
                        if max_pos_evidences and len(answering_evidences) <= max_pos_evidences:
                            if hit:
                                aCount += 1
                                newConv["questions"].append(turn)
                        turn["answer_presence_per_src"] = {
                            evidence["source"]: 1 for evidence in answering_evidences
                        }
                    answerable_convs.append(newConv)

                    fp2.write(json.dumps(newConv))
                    fp2.write("\n")
                   
                    # write conversation to file
                    fp.write(json.dumps(conversation))
                    fp.write("\n")
                print("total number of answerable refs: ", aCount)

                # accumulate results
                c_answer_presences = [turn["answer_presence"] for turn in conversation["questions"]]
                answer_presences += c_answer_presences
                for turn in conversation["questions"]:
                    answer_presence_per_src = turn["answer_presence_per_src"]
                    # add per source answer presence
                    for src, ans_presence in answer_presence_per_src.items():
                        source_to_ans_pres[src] += ans_presence
                    # aggregate overall answer presence for validation
                    if len(answer_presence_per_src.items()):
                        source_to_ans_pres["all"] += 1

        # print results
        res_path = output_path.replace(".jsonl", ".res")
        with open(res_path, "w") as fp:
            avg_answer_presence = sum(answer_presences) / len(answer_presences)
            fp.write(f"Avg. answer presence: {avg_answer_presence}\n")
            answer_presence_per_src = {
                src: (num / len(answer_presences)) for src, num in source_to_ans_pres.items()
            }
            fp.write(f"Answer presence per source: {answer_presence_per_src}")


        if self.config.get("ref_per_epoch", False) and "train" in output_path:
            for i in range(self.config.get("ref_num", 5)):
                ref_path = output_path.replace(".jsonl", "_" + str(i)+"ref.jsonl")
                dataPerRef = []
                aIntent = 0
                for conv in answerable_convs:
                    if len(conv["questions"]) == 0:
                        #print("len 0: ", conv["conversation_id"])
                        continue
                    aIntent += 1
                    newConv = dict()
                    newConv["conversation_id"] = conv["conversation_id"]
                    idx = i%len(conv["questions"])
                    #print("idx: ", idx, "len: ", len(conv["questions"]))
                    newConv["questions"] = [conv["questions"][idx]]
                    dataPerRef.append(newConv)
                print("answerableIntent: ", aIntent, "len data per ref: ", len(dataPerRef))
                with open(ref_path, "w") as fp:
                    for entry in dataPerRef:
                        fp.write(json.dumps(entry))
                        fp.write("\n")

        # log
        self.logger.info(f"Done with processing: {input_path}.")


    def inference_on_data(self, input_data, sources=("kb", "text", "table", "info"), train=False):
        """Run ERS on given data."""
        input_turns = [turn for conv in input_data for turn in conv["questions"]]
        self.inference_on_turns(input_turns, sources, train)
        return input_data

    def inference_on_turns(self, input_turns, sources=("kb", "text", "table", "info"), train=False):
        """Run ERS on given turns."""
        for turn in tqdm(input_turns):
            if "top_evidences" in turn:
                continue
            self.inference_on_turn(turn, sources, train)
        return input_turns

    def inference_on_turn(self, *args):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def store_cache(self):
        pass

    def load(self):
        pass
