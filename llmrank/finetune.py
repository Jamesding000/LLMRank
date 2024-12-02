import importlib
import argparse
import os
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from model import llmranker  # Ensure you import the module's parent package
from model.llmranker import LLMRanker  # Re-import the specific class
from dataset import LLMRankerDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, DataLoader
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)

import logging
from dataclasses import dataclass, field

import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_MAP = {
    "llama3": "Llama-3.2-1B",  #"meta-llama/Llama-3.2-1B"
    "gpt-neo":"gpt-neo-1.3B" #"EleutherAI/gpt-neo-1.3B"
}

model_name = "llama3"
dataset_name = 'ml-1m-full'

def finetune(model_name, dataset_name, **kwargs):
    
    def preprocess_batch(batch):
        """
        Tokenizes the text batch for fine-tuning.
        """
        inputs = tokenizer(
            batch,                # Concatenated input + target strings
            padding=True,         # Pad to the longest sequence in the batch
            truncation=True,      # Truncate sequences exceeding the max length
            max_length=512,       # Adjust based on model's context length
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Create labels for loss calculation, masking input tokens
        labels = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": attention_mask,
            "labels": labels,  # Loss is computed on these
        }
        
    props = [f'props/{model_name}.yaml', f'props/{dataset_name}.yaml', 'openai_api.yaml', 'props/overall.yaml']
    print(props)
    # model_class = get_model(model_name)

    model_class = LLMRanker
    model_path = os.path.join('./models', MODEL_MAP[model_name])

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    
    dataset = SequentialDataset(config)
    
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    movie_dataset = LLMRankerDataset(config, dataset)
    movie_dataset.build(test_data)

    # Define split ratios
    train_ratio = 0.9

    # Calculate sizes of each split
    total_size = len(movie_dataset)
    train_size = int(total_size * train_ratio)
    valid_size = total_size - train_size  # Ensure all samples are used

    # Perform the split
    train_dataset, valid_dataset = random_split(
        movie_dataset, 
        [train_size, valid_size], 
        generator=torch.Generator().manual_seed(42)  # Seed for reproducibility
    )

    print('Finetune dataset example \n',train_dataset[0])

    # # Create DataLoaders
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    @dataclass
    class ModelConfig:
        model_to_train: str = field(default="models/gpt-neo-1.3B")
        seq_len: int = field(default=512)
        attention_type: str = field(default="flash_attention_2")

    config_file = os.path.join(f'configs/{model_name}.json')
    parser = HfArgumentParser((ModelConfig, TrainingArguments))
    model_config, training_args = parser.parse_json_file(json_file=config_file)

    logger.info(f"Base model: {model_config.model_to_train}")
    logger.info(f"Saving to: {training_args.output_dir}")

    llm = AutoModelForCausalLM.from_pretrained(model_config.model_to_train, 
                                                torch_dtype=torch.bfloat16, 
                                                attn_implementation=model_config.attention_type,
                                                trust_remote_code=True
                                            ).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_to_train)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=llm,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=preprocess_batch,
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    trainer.save_model(training_args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="Rank", help="model name")
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-g', type=str, default='0', help='Visible GPU devices')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.m, args.d, gpu_id=args.g)
