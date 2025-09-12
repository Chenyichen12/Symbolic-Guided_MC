import json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random
import re
from datetime import datetime
from os.path import isfile
from vllm import LLM, SamplingParams


P_LIST = ["history", "finance", "diving", "sport", "writing", "gardening", "shopping", "marketing", "recruiting", "medical", "music"]


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--dataset", type=str, default="LogicAsker/exp_len_3.json")
    parser.add_argument("--persona", choices=["logic", "vanilla", "n", "random"], default="logic")
    parser.add_argument("--sample_rate", type=int, default=5)
    parser.add_argument("--vanilla", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    model_prefix = args.model_name.split(":")[0]

    if args.persona == "vanilla":
        if "logicasker" in args.data_path.lower():
            example_path = "prompts/LogicAsker_v_react_example.txt"
            instruction_path = "prompts/LogicAsker_v_react_instruction.txt"
            save_prefix = "logicasker_vanilla"
        elif "folio" in args.data_path.lower():
            example_path = "prompts/FOLIO_v_react_example.txt"
            instruction_path = "prompts/FOLIO_v_react_instruction.txt"
            save_prefix = "folio_vanilla"
        else:
            print("Dataset not supported")
            exit(0)
    elif args.persona == "logic":
        if "logicasker" in args.data_path.lower():
            example_path = "prompts/LogiAsker_react_example.txt"
            instruction_path = "prompts/LogicAsker_instruction.txt"
            save_prefix = "logicasker"
        elif "folio" in args.data_path.lower():
            example_path = "prompts/FOLIO_react_example.txt"
            instruction_path = "prompts/FOLIO_instruction.txt"
            save_prefix = "folio"
        else:
            print("Dataset not supported")
            exit(0)
    
    with open(example_path, 'r') as f:
        prompt_example1 = f.read().strip()

    with open(instruction_path, 'r') as f:
        instruction_prompt = f.read()
    
    if "FOLIOv2" in args.dataset:
        save_prefix = "folio_v2"

    
    with open(f"symbcot_data/{args.dataset}", 'r') as f:
        data_list = json.load(f)

    if args.persona == "logic":
        system_prompt = "You are a logic expert who prefer to reason using symbols instead of natural language. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "vanilla":
        system_prompt = "You are a helpful assistant. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "n":
        system_prompt = "You are not a logic expert. You prefer to reason using natural language instead of symbols. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "random":
        sys_persona_id_list = np.random.randint(0, len(P_LIST), len(data_list)).tolist()

    with open("prompts/logic_rules.txt", 'r') as f:
        logic_rules = f.read().strip()

    if data_list[0].get("question") is None or data_list[0].get("question") == "":
        seed_inputs = [f"{instruction_prompt}=======Example=======\n\n{prompt_example1}\n\n=======Input=======\n\nContext:{d['context']}" for d in data_list]
    else:
        seed_inputs = [f"{instruction_prompt}=======Example=======\n\n{prompt_example1}\n\n=======Input=======\n\nContext:{d['context']}\n{d['question']}" for d in data_list]

    out_path = f"mc_react_trajectories/{save_prefix}_{model_prefix}_{args.persona}_{args.seed}.json"

    if isfile(out_path):
        with open(out_path, 'r') as f:
            output_file = json.load(f)
    else:
        output_file = {}

    sampling_params = SamplingParams(
          temperature=1.0,
          max_tokens=2048,
      )

    llm = LLM(model=args.model_name, seed=args.seed, max_model_len=8192)


    for i in tqdm(range(len(output_file), len(seed_inputs))):
        if args.persona == "random":
            system_prompt = f"You are a {P_LIST[sys_persona_id_list[i]]} expert. You would follow the instructions in the prompt and return the reasoning steps."

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": seed_inputs[i]}
        ]

        conversations = [conversation for _ in range(args.sample_rate)]

        if "Qwen3" in args.model_name:
            outputs = llm.chat(conversations, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False})
        else:
            outputs = llm.chat(conversations, sampling_params=sampling_params)

        samples = [output.outputs[0].text for output in outputs]
        
        question_text = data_list[i]['context']
        if data_list[i].get('question') is not None and data_list[i]['question'] != "":
            question_text += "\n"
            question_text += data_list[i]['question']

        if data_list[i].get('options') is not None and len(data_list[i]['options']) > 0:
            question_text += "\nOptions:\n"
            this_option = '\n'.join(data_list[i]["options"])
            question_text += this_option
            
        gold_answer = data_list[i]["answer"]

        data_id = data_list[i]["id"]
        output_file[data_id] = {
          "question": question_text,
          "gold_answer": gold_answer,
          "samples": samples
        }


        with open(out_path, 'w') as f:
            json.dump(output_file, f, indent=4)
      
    end_time = datetime.now()
    print(f"Finishing script at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    print(f"Time spent: {time_diff}")
