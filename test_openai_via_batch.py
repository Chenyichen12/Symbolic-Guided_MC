"""
generating a batch API file as the following format:
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

"""
import json
from argparse import ArgumentParser
from os.path import isfile
import numpy as np


def build_prompt(instruction_prompt: str, context: str, example: str, question: str = "") -> str:
    prompt = f"{instruction_prompt}=======Example=======\n\n{example}\n\n=======Input=======\n\nContext:{context}\n{question}"
    return prompt
   

P_LIST = ["history", "finance", "diving", "sport", "writing", "gardening", "shopping", "marketing", "recruiting", "medical", "music"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5")
    parser.add_argument("--persona", choices=["logic", "vanilla", "n", "random"], default="vanilla")
    parser.add_argument("--data_path", type=str, default="symbcot_data/LogicAsker/tes3380.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.persona == "vanilla":
        if "logicasker" in args.data_path.lower():
            example_path = "prompts/LogicAsker_v_react_example.txt"
            instruction_path = "prompts/LogicAsker_v_react_instruction.txt"
            this_dataset_name = "logicasker"
        elif "folio" in args.data_path.lower():
            example_path = "prompts/FOLIO_v_react_example.txt"
            instruction_path = "prompts/FOLIO_v_react_instruction.txt"
            this_dataset_name = "folio"
        else:
            print("Dataset not supported")
            exit(0)
    elif args.persona == "logic":
        if "logicasker" in args.data_path.lower():
            example_path = "prompts/LogiAsker_react_example.txt"
            instruction_path = "prompts/LogicAsker_instruction.txt"
            this_dataset_name = "logicasker"
        elif "folio" in args.data_path.lower():
            example_path = "prompts/FOLIO_react_example.txt"
            instruction_path = "prompts/FOLIO_instruction.txt"
            this_dataset_name = "folio"
        else:
            print("Dataset not supported")
            exit(0)
        

    with open(example_path, 'r') as f:
        prompt_example1 = f.read().strip()

    with open(instruction_path, 'r') as f:
        instruction_prompt = f.read()
  
    with open(args.data_path, 'r') as f:
        data_list = json.load(f)

    if args.persona == "logic":
        system_prompt = "You are a logic expert who prefer to reason using symbols instead of natural language. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "vanilla":
        system_prompt = "You are a helpful assistant. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "n":
        system_prompt = "You are not a logic expert. You prefer to reason using natural language instead of symbols. You would follow the instructions in the prompt and return the reasoning steps."
    elif args.persona == "random":
        sys_persona_id_list = np.random.randint(0, len(P_LIST), len(data_list)).tolist()

    if data_list[0].get("question") is not None:
        need_question = True
    else:
        need_question = False

    model_prefix = args.model_name.split(":")[0]
    save_path = f"batch_api/openai_batch_api_{model_prefix}_{args.persona}_{this_dataset_name}_{args.seed}.jsonl"

    out_str = ""
        

    for i in range(len(data_list)):
        if args.persona == "random":
            system_prompt = f"You are a {P_LIST[sys_persona_id_list[i]]} expert. You would follow the instructions in the prompt and return the reasoning steps."

        if need_question:
            this_input = build_prompt(instruction_prompt, data_list[i]["context"], prompt_example1, data_list[i]["question"])
        else:
            this_input = build_prompt(instruction_prompt, data_list[i]["context"], prompt_example1)
        
        data_id = data_list[i]["id"]

        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": this_input},
                ]

        this_call = {
            "custom_id": data_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model_name,
                "messages": messages,
                "temperature": 1.0,
                "seed": args.seed
            }
        }

        out_str += json.dumps(this_call) + "\n"
    
    out_str = out_str[:-1]  # remove the last newline character
    with open(save_path, 'w') as f:
        f.write(out_str)

    print(f"Batch API file saved to {save_path}")
    

        