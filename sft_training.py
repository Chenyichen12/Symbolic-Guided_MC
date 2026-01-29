"""
Example training script for a process reward model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
import torch
import os
import json
from datetime import datetime
import yaml
from dataclasses import dataclass, field
import torch.nn.functional as F
from transformers.utils.generic import PaddingStrategy
from argparse import ArgumentParser
from os.path import join, isfile
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
import numpy as np


os.environ["WANDB_PROJECT"]="Stanage-sft-training"


def prepare_dataset(raw_dataset):
    data_list = []
    if type(raw_dataset) == list:
        for q_dict in raw_dataset:
            message = [
                {"role":"user", "content":q_dict["question"]},
                {"role":"assistant", "content":q_dict["response"]}
            ]
            data_list.append(message)
    else:
        df = raw_dataset.to_pandas()
        for i in range(len(df)):
            message = [
                {"role":"user", "content":df.iloc[i]["question"]},
                {"role":"assistant", "content":df.iloc[i]["response"]}
            ]
            data_list.append(message)

    return {"messages": data_list}


def prepare_eval_dataset(raw_dataset):
    input_list = []
    label_list = []

    if len(raw_dataset[0].get("options")) > 0:
        for data_dict in raw_dataset:
            this_option = '\n'.join(data_dict["options"])
            this_contenct = f"{data_dict['context']}\n{data_dict['question']}\nOptions:\n{this_option}"
            this_message = [
                {"role":"user", "content": this_contenct},
            ]
            input_list.append(this_message)
            label_list.append(data_dict["answer"])
    else:
        for data_dict in raw_dataset:
            this_contenct = f"{data_dict['context']}\n{data_dict['question']}"
            this_message = [
                {"role":"user", "content": this_contenct},
            ]
            input_list.append(this_message)
            label_list.append(data_dict["answer"])

    return {"messages": input_list, "labels": label_list}


def main(args):
    start_time = datetime.now()
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    with open(args.config_path, "r") as f:
        trainer_config = yaml.safe_load(f)


    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize tokenizer and model

    if "qwen" in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right", use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    model = AutoModelForCausalLM.from_pretrained(
            args.model_name, num_labels=1, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    model.config.use_cache = False ########
    model.config.pad_token_id = tokenizer.pad_token_id


    # Create and process dataset
    if "XingweiT/" in args.data_path:
        raw_dataset = load_dataset(args.data_path, split="train")
    else:
        with open(args.data_path, 'r') as f:
            raw_dataset = json.load(f)

    sft_dataset = Dataset.from_dict(prepare_dataset(raw_dataset)).shuffle(seed=args.seed)

    if len(args.eval_path) > 0:
        with open(args.eval_path, 'r') as f:
            eval_dataset = json.load(f)

    if len(args.eval_path) > 0:
        eval_sft_dataset = Dataset.from_dict(prepare_dataset(eval_dataset))


    training_args = SFTConfig(**trainer_config)


    # Initialize and train
    if len(args.eval_path) == 0:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=sft_dataset,
            processing_class=tokenizer,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=sft_dataset,
            eval_dataset=eval_sft_dataset,
            processing_class=tokenizer,
        )

    # Wrap the training in a try-except block
    try:
        if args.resume:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        # Clean up any resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        raise e


    end_time = datetime.now()
    print(f"Finishing script at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    print(f"Time spent: {time_diff}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--config-path", type=str, default="config/sft_config_qwen.yaml")
    parser.add_argument("--data-path", type=str, default="XingweiT/SymbReAct-trace-sft")
    parser.add_argument("--eval-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()

    main(args)