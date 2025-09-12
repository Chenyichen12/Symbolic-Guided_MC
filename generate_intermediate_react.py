from reasoners.utils import OllamaModel
import json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from datetime import datetime
from os.path import isfile
from utils import get_steps


CONTINUE_PROMPT = "I have wrote the first part of the reasoning path. Please continue the reasoning path."

P_LIST = ["history", "finance", "diving", "sport", "writing", "gardening", "shopping", "marketing", "recruiting", "medical", "music"]


def process_output(text: str):
    react_list = text.split('\n')
    for i in range(len(react_list)):
      if react_list[i].startswith("Action:") and i>0 and not react_list[i-1].startswith("Thought:"):
        react_list[i] = '\n\n' + react_list[i]
      
    out_str = '\n'.join(react_list)
    out_str.replace('\nThought:', '\n\nThought:')

    return out_str


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5:7b-instruct-fp16")
    parser.add_argument("--dataset", type=str, default="data/LogicAsker/test.json")
    parser.add_argument("--persona", choices=["logic", "vanilla", "n", "random"], default="logic")
    parser.add_argument("--sample_rate", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ollama_hanlder = OllamaModel(args.model_name, ['------'], -1, host="http://localhost:11434")


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


    if data_list[0].get("question") is None or data_list[0].get("question") == "":
        seed_inputs = [f"{instruction_prompt}=======Example=======\n\n{prompt_example1}\n\n=======Input=======\n\nContext:{d['context']}" for d in data_list]
    else:
        seed_inputs = [f"{instruction_prompt}=======Example=======\n\n{prompt_example1}\n\n=======Input=======\n\nContext:{d['context']}\n{d['question']}" for d in data_list]

    out_path = f"prm_samples/{save_prefix}_mc_{model_prefix}_{args.persona}_{args.seed}.json"

    if isfile(out_path):
        with open(out_path, 'r') as f:
          output_file = json.load(f)
    else:
        output_file = {
          "raw_outputs": []
        }


    for i in tqdm(range(len(output_file["raw_outputs"]), len(seed_inputs))):
        if args.persona == "random":
          system_prompt = f"You are a {P_LIST[sys_persona_id_list[i]]} expert. You would follow the instructions in the prompt and return the reasoning steps."

        output = ollama_hanlder.generate(seed_inputs[i], system_prompt=system_prompt, seed=args.seed)

        steps = get_steps(output)

        sampled_output_list = []
        for j in range(len(steps)-1):
            if "Action: Finish" in steps[j]:
                break

            sample_input = seed_inputs[i] + "\n\n" + CONTINUE_PROMPT + "\n\n" + "\n\n".join(steps[:j+1])
            samples_at_this_step = []
            for k in range(1, args.sample_rate+1):
                sample_output = ollama_hanlder.generate(sample_input, system_prompt=system_prompt, seed=j*args.seed+k+10)
                samples_at_this_step.append(sample_output)
            sampled_output_list.append(samples_at_this_step)

        question_text = "Problem: \n\n"
        question_text += data_list[i]['context']
        if data_list[i].get('question') is not None and data_list[i]['question'] != "":
            question_text += "\n"
            question_text += data_list[i]['question']

        if data_list[i].get('options') is not None and len(data_list[i]['options']) > 0:
            question_text += "\nOptions:\n"
            this_option = '\n'.join(data_list[i]["options"])
            question_text += this_option
        question_text += "\n\n"
        
        gold_answer = data_list[i]["answer"]
          
        this_dict = {
            "seed_reasoning_path": steps,
            "sampled_reasoning_paths": sampled_output_list,
            "gold_answer": gold_answer,
            "question_text": question_text
          }
        output_file["raw_outputs"].append(this_dict)

        with open(out_path, 'w') as f:
            json.dump(output_file, f, indent=4)

      
    end_time = datetime.now()
    print(f"Finishing script at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    print(f"Time spent: {time_diff}")
