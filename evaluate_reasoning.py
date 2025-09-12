from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random
import re
from datetime import datetime
from os.path import isfile
import numpy as np
import torch
from utils import parse_leaf_node_value


SIMPLE_INSTRUCTION = 'Solve a question answering task by reasoning based on existing facts, then Finish with your answer. The last line should be Finish [answer] which returns the answer (please choose from True, False, or Uncertain) and finishes the task. You will be given context that you should use to help you answer the question. If it is possible to infer the statement, please answer "True". If it is possible to infer the negation of the statement, please answer "False". If the statement cannot be proofed or disproofed, please answer "Uncertain".'


SIMPLE_INSTRUCTION_BINARY = 'Solve a question answering task by reasoning based on existing facts, then Finish with your answer. The last line should be Finish [answer] which returns the answer (please choose from True or False) and finishes the task. You will be given context that you should use to help you answer the question.'


P_LIST = ["history", "finance", "diving", "sport", "writing", "gardening", "shopping", "marketing", "recruiting", "medical", "music"]


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/llama3.1-8B-dpo/checkpoint-1970")
    parser.add_argument("--dataset", type=str, default="data/LogicAsker/test.json")
    parser.add_argument("--persona", choices=["logic", "vanilla", "n", "random"], default="logic")
    parser.add_argument("--sample_rate", type=int, default=5)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--react", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    if "FOLIO" in args.dataset:
        with open("prompts/FOLIO_react_example.txt", 'r') as f:
            prompt_example1 = f.read().strip()
        save_prefix = "folio"
        if args.react:
            save_prefix += "_react"
            with open("prompts/FOLIO_v_react_instruction.txt", 'r') as f:
                instruction = f.read()
        else:
            instruction = SIMPLE_INSTRUCTION
    elif "LogicAsker" in args.dataset or "Phemeplus" in args.dataset or "VitC" in args.dataset or "Climate_fever" in args.dataset:
        with open("prompts/LogiAsker_react_example.txt", 'r') as f:
            prompt_example1 = f.read().strip()
        save_prefix = args.dataset.split("/")[1].lower()

        if args.react:
            save_prefix += "_react"
            with open("prompts/LogicAsker_v_react_instruction.txt", 'r') as f:
                instruction = f.read()
        else:
            instruction = SIMPLE_INSTRUCTION_BINARY
    else:
        print("Dataset not supported")
        exit(0)

  
    with open(args.dataset, 'r') as f:
        data_list = json.load(f)


    if args.persona == "logic":
        system_prompt = "You are a logic expert who prefer to reason using symbols instead of natural language. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "vanilla":
        system_prompt = "You are a helpful assistant. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "n":
        system_prompt = "You are not a logic expert. You prefer to reason using natural language instead of symbols. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "random":
        sys_persona_id_list = np.random.randint(0, len(P_LIST), len(data_list)).tolist()


    model_prefix = args.model_path.replace("/", "_")
    data_prefix = args.dataset.split("/")[-1].split(".")[0]


    if args.zero_shot:
        if data_list[0].get("question") is not None and len(data_list[0].get("question")) > 0:
            seed_inputs = [f"{instruction}=======Input=======\n\nContext:{d['context']}\n{d['question']}\n"+'\n'.join(d['options']) for d in data_list]
        else:
            seed_inputs = [f"{instruction}=======Input=======\n\nContext:{d['context']}" for d in data_list]
        out_path = f"eval_output/{save_prefix}_{data_prefix}_{model_prefix}_zero.json"
    else:
        if data_list[0].get("question") is not None and len(data_list[0].get("question")) > 0:
            seed_inputs = [f"{instruction}=======Example=======\n\n{prompt_example1}\n\n=======Input=======\n\nContext:{d['context']}\n{d['question']}\n"+'\n'.join(d['options']) for d in data_list]
        else:
            seed_inputs = [f"{instruction}=======Example=======\n\n{prompt_example1}\n\n=======Input=======\n\nContext:{d['context']}" for d in data_list]
        out_path = f"eval_output/{save_prefix}_{data_prefix}_{model_prefix}.json"


    if isfile(out_path):
        with open(out_path, 'r') as f:
            output_file = json.load(f)
    else:
        output_file = {}


    for i in tqdm(range(len(output_file), len(seed_inputs))):
        if args.persona == "random":
            system_prompt = f"You are a {P_LIST[sys_persona_id_list[i]]} expert. You would follow the instructions in the prompt and return the reasoning steps."

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": seed_inputs[i]}
        ]
        model_input = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(model_input, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, num_return_sequences=1, do_sample=True, temperature=0.3, top_k=50, top_p=0.95, num_beams=1, early_stopping=True, max_new_tokens=2048)
        decoded_output = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        question_text = data_list[i]['context']
        gold_answer = data_list[i]["answer"]



        matched_correct, predicted_label = parse_leaf_node_value(decoded_output, gold_answer)

        data_id = data_list[i]["id"]
        output_file[data_id] = {
          "question": question_text,
          "gold_answer": gold_answer,
          "output": decoded_output,
          "matched_correct": matched_correct,
          "predicted_label": predicted_label
        }


        with open(out_path, 'w') as f:
            json.dump(output_file, f, indent=4)
    
    
    accuracy = np.mean([output_file[k]["matched_correct"] for k in output_file])
    print(f"Accuracy: {accuracy}")
      
    end_time = datetime.now()
    print(f"Finishing script at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    print(f"Time spent: {time_diff}")
