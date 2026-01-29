from datasets import Dataset, load_dataset
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import yaml
from argparse import ArgumentParser
import os
import torch


os.environ["WANDB_PROJECT"]="dpo-training"


def create_training_examples(raw_dataset):
    train_samples = []
    for this_dict in raw_dataset:
        this_sample = {
            "chosen": this_dict["positive"],
            "rejected": this_dict["negative"]
        }
        train_samples.append(this_sample)
    return train_samples


def main(args):
    if "XingweiT/" in args.data_path:
        train_dataset = load_dataset(args.data_path, split="train")
    else:
        with open(args.data_path, "r") as f:
            raw_dataset = json.load(f)
        train_dataset = Dataset.from_list(create_training_examples(raw_dataset)).shuffle(seed=42)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config.get("bf16", False):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    if args.pipeline_parallel:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map="auto", use_flash_attention_2=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, use_flash_attention_2=True)
    
    if "qwen" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right", use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    training_args = DPOConfig(**config)
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, default="XingweiT/SymbReAct-trace-dpo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--config-path", type=str, default="config/dpo_config_qwen.yaml")
    parser.add_argument("--pipeline-parallel", action="store_true")
    args = parser.parse_args()
    main(args)