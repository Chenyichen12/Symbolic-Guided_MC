import json
import re
from argparse import ArgumentParser
from utils import parse_leaf_node_value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file with gold answers")
    parser.add_argument("--response_path", type=str, required=True, help="Path to the model output file")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    answer_dict = {item['id']: item['answer'] for item in data if 'id' in item and 'answer' in item}

    output_dict = {}
    correct = []
    with open(args.response_path, 'r') as f:
        for line in f.readlines():
            response_data = json.loads(line.strip())
            response_text = response_data["response"]["body"]["choices"][0]["message"]["content"]
            data_id = response_data["custom_id"]
            gold_answer = answer_dict.get(data_id)

            if gold_answer:
                matched_correct, predicted_label = parse_leaf_node_value(response_text, gold_answer)
                correct.append(matched_correct)

                output_dict[data_id] = {
                   "matched_correct": matched_correct,
                   "predicted_label": predicted_label,
                   "response_text": response_text,
                   "gold_answer": gold_answer
                }
    
    file_name = args.response_path.split("/")[-1]
    with open(f"eval_output/{file_name}", 'w') as f:
        json.dump(output_dict, f, indent=4)
    
    print(f"Evaluation results saved to eval_output/{file_name}")

    accuracy = sum(correct) / len(correct) if correct else 0
    print(f"Accuracy: {accuracy}")
