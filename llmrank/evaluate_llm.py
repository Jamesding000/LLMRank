import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from utils import get_model
from trainer import SelectedUserTrainer
from model.llmranker import LLMRanker
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_MAP = {
    "llama3": "Llama-3.2-1B",  #"meta-llama/Llama-3.2-1B"
    "gpt-neo":"gpt-neo-1.3B" #"EleutherAI/gpt-neo-1.3B"
}

model_name = "llama3"
dataset_name = 'ml-1m-full'

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_to_train: str = field(default="models/gpt-neo-1.3B")
    seq_len: int = field(default=512)
    attention_type: str = field(default="flash_attention_2")

def evaluate(model_name, dataset_name, model_path, **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', f'props/{dataset_name}.yaml', 'openai_api.yaml', 'props/overall.yaml']
    print(props)
    # model_class = get_model(model_name)

    model_class = LLMRanker

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = SequentialDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    config_file = os.path.join(f'configs/{model_name}.json')
    parser = HfArgumentParser((ModelConfig, TrainingArguments))
    model_config, _ = parser.parse_json_file(json_file=config_file)

    logger.info(f"Loaded model: {model_config.model_to_train}")

    llm = AutoModelForCausalLM.from_pretrained(model_path, 
                                                torch_dtype=torch.bfloat16, 
                                                attn_implementation=model_config.attention_type,
                                                trust_remote_code=True
                                            ).to(config['device'])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # model loading and initialization
    model = LLMRanker(config, train_data.dataset, llm, tokenizer).to(config['device'])

    logger.info(model)

    # trainer loading and initialization
    trainer = SelectedUserTrainer(config, model, dataset)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    output_res = []
    for u, v in test_result.items():
        output_res.append(f'{v}')
    logger.info('\t'.join(output_res))

    return config['model'], config['dataset'], {
        'valid_score_bigger': config['valid_metric_bigger'],
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="llama3", help="model name")
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-p', type=str, default='models/Llama-3.2-1B-sft', help='model path')
    args, unparsed = parser.parse_known_args()
    print(args)

    evaluate(args.m, args.d, model_path=args.p)
