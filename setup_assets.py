import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

def download_assets():
    print("Downloading assets for SymbReAct-trace...")

    # 1. Download Datasets
    datasets = [
        "XingweiT/SymbReAct-trace-sft",
        "XingweiT/SymbReAct-trace-dpo"
    ]
    
    for ds_name in datasets:
        print(f"Downloading dataset: {ds_name}")
        try:
            load_dataset(ds_name, split="train")
            print(f"Successfully downloaded {ds_name}")
        except Exception as e:
            print(f"Error downloading {ds_name}: {e}")

    # 2. Download Models
    # We will skip downloading the large model weights in this script to save time.
    # The training script or 'huggingface-cli download' can handle it.
    print("Skipping model download in this script. Use 'huggingface-cli download Qwen/Qwen2.5-7B-Instruct' or run the training script to download automatically.")

if __name__ == "__main__":
    download_assets()
