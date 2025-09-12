"""
Example inference script for a process reward model.
"""

import json
from argparse import ArgumentParser
import numpy as np
import re
from utils import parse_leaf_node_value


'''
Data format:
{
    "data_id": {
        "question": "question",
        "samples": [
            ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10"],
            ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10"],
            ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10"],
        ]
    }
'''



def main(args):
    with open(args.input_path, 'r') as f:
        sampled_data = json.load(f)
    
    training_samples = []
    false_positives = 0
    total = 0
    for data_id in sampled_data:
        for i in range(len(sampled_data[data_id]["samples"])):
            al_correct = True
            for score in sampled_data[data_id]["sample_scores"][i]:
                if score <= 0.5:
                    al_correct = False
                    break
            
            matched_correct, _ = parse_leaf_node_value(sampled_data[data_id]["samples"][i][-1], sampled_data[data_id]["gold_answer"])

            if al_correct:
               total += 1

            if matched_correct == 0:
                if al_correct:
                   false_positives += 1
                al_correct = False
               
            

            if al_correct:
                training_samples.append({
                    'question': sampled_data[data_id]["question"],
                    'response': '\n\n'.join(sampled_data[data_id]["samples"][i]),
                })
    print("FP rate: ", false_positives / total)
    print("Total before filtering with gold labels: ", total)
    print(len(training_samples))
    with open(args.input_path.replace("dpo_pairs/", "sft_samples/"), 'w') as f:
        json.dump(training_samples, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="dpo_pairs/0329.json")
    args = parser.parse_args()

    main(args)
