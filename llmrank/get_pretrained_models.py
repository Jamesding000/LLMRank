
import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_MAP = { # map model id to local directory
    "meta-llama/Llama-3.2-1B": "./models/Llama-3.2-1B",
    "EleutherAI/gpt-neo-1.3B": "./models/gpt-neo-1.3B"
}

for model_id, local_dir in MODEL_MAP.items():
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Save the model and tokenizer to the local directory for future use
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
