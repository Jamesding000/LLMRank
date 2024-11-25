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

MODEL_MAP = {
    "llama3": "meta-llama/Meta-Llama-3-8B"
}

def train(model_name, dataset_name, **kwargs):
    
        # configurations initialization
    props = [f'props/{model_name}.yaml', f'props/{dataset_name}.yaml', 'openai_api.yaml', 'props/overall.yaml']
    print(props)
    model_class = get_model(model_name)

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
    
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
        
    # model loading and initialization
    model = LLMRanker(config, train_data.dataset, llm, tokenizer).to(config['device'])

    # Load pre-trained model
    # if pretrained_file != '':
    #     checkpoint = torch.load(pretrained_file)
    #     logger.info(f'Loading from {pretrained_file}')
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     model.load_other_parameter(checkpoint.get("other_parameter"))

    logger.info(model)

    # trainer loading and initialization
    trainer = SelectedUserTrainer(config, model, dataset)
    
    # Define the train function help. Load the model and dataset
    # prepare data collator and construct trainer
    # Trainer the model
    # Save the modle
    pass 


def evaluate(model_name, dataset_name, **kwargs):
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
    
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
    
    # model loading and initialization
    model = LLMRanker(config, train_data.dataset, llm, tokenizer).to(config['device'])

    # Load pre-trained model
    # if pretrained_file != '':
    #     checkpoint = torch.load(pretrained_file)
    #     logger.info(f'Loading from {pretrained_file}')
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     model.load_other_parameter(checkpoint.get("other_parameter"))

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
    parser.add_argument("-m", type=str, default="Rank", help="model name")
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    args, unparsed = parser.parse_known_args()
    print(args)

    evaluate(args.m, args.d, pretrained_file=args.p)
