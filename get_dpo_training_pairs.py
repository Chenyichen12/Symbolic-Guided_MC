"""
Example inference script for a process reward model.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
from sklearn.metrics import accuracy_score, f1_score
from argparse import ArgumentParser
import numpy as np
from os import listdir
from os.path import isfile, join
from utils import get_steps


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
    model_middle = args.model_path.split('/')[1]
    save_path1 = args.output_path.replace(".json", f"_{model_middle}.json")
    save_path2 = args.output_path.replace(".json", f"_{model_middle}_dpo_pairs.json")
    if isfile(save_path1):
        with open(save_path1, 'r') as f:
            sampled_data = json.load(f)
        with open(save_path2, 'r') as f:
            dpo_pairs = json.load(f)
    else:
        sampled_data = {}
        dpo_pairs = []
        for file_name in listdir(args.data_dir):
            if not file_name.endswith(".json"):
                continue
            with open(join(args.data_dir, file_name), 'r') as f:
                this_dict = json.load(f)
            
            for data_id in this_dict:
                if data_id in sampled_data:
                    sampled_data[data_id]["samples"] += this_dict[data_id]["samples"]
                else:
                    sampled_data[data_id] = this_dict[data_id]

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()


    for data_id in sampled_data:
        if sampled_data.get("sample_scores") is not None:
            continue

        al_scores = []
        accumulated_scores = []
        temp_response_list = []
        question = sampled_data[data_id]["question"]
        for trajectory in sampled_data[data_id]["samples"]: # when forget to split the steps in the trajectory file, apply get_steps here
            trajectory_score = []
            eval_input = ""
            the_steps = get_steps(trajectory)
            for i, step in enumerate(the_steps):
                eval_input += f"Step {i+1}:\n\n{step}\n"
                message = [
                    {"role":"user", "content":question},
                    {"role":"assistant", "content":eval_input}
                ]
                model_input = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False)
                inputs = tokenizer(model_input, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.sigmoid(outputs.logits)
                trajectory_score.append(probs.item())
            al_scores.append(trajectory_score)

            cumulative_product = np.cumprod(trajectory_score)
            accumulated_scores.append(cumulative_product[-1])

            temp_response_list.append(eval_input)
        
        for i in range(len(accumulated_scores)):
            for j in range(len(accumulated_scores)):
                if i != j and accumulated_scores[i] - accumulated_scores[j] > args.threshold:
                    dpo_pairs.append(
                        {
                            "question_id": data_id,
                            "question": question,
                            "positive": temp_response_list[i],
                            "negative": temp_response_list[j]
                        }
                    )

        sampled_data[data_id]["sample_scores"] = al_scores

        with open(save_path1, 'w') as f:
            json.dump(sampled_data, f, indent=4)
        
        with open(save_path2, 'w') as f:
            json.dump(dpo_pairs, f, indent=4)

    print(f"Number of DPO pairs: {len(dpo_pairs)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="trajetory_samples")
    parser.add_argument("--output_path", type=str, default="dpo_pairs/0329.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    main(args)