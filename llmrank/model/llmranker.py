import os.path as osp
import torch
import openai
import time
import asyncio
import numpy as np
from tqdm import tqdm
import pylcs
import html
import replicate
from recbole.model.abstract_recommender import SequentialRecommender
# from vllm import LLM, SamplingParams

from utils import dispatch_openai_requests, dispatch_single_openai_requests

import json
def save_responses_to_file(responses, file_path="openai_responses.json"):
    """
    Save OpenAI responses to a file after making them JSON serializable.

    Args:
        responses: The OpenAI responses to save.
        file_path: The file path to save the responses (default: 'openai_responses.json').
    """
    # Extract serializable parts of responses
    serializable_responses = [
        response.to_dict() if hasattr(response, "to_dict") else response
        for response in responses
    ]
    
    with open(file_path, "w") as file:
        json.dump(serializable_responses, file, indent=4)
    
    print(f"Responses saved to {file_path}")

def save_prompt_list_to_file(prompt_list, file_path="prompts.json"):
    """
    Save a list of strings to a file by dumping it as JSON.

    Args:
        prompt_list: List of strings to save.
        file_path: Path to the file (default: 'prompts.json').
    """
    with open(file_path, "w") as file:
        json.dump(prompt_list, file, indent=4)  # Save as JSON array
    print(f"Prompts saved to {file_path}")


# We don't change the LLMRanker class, which is a wrapper class of the llm. We train the llm seperately.
# We construct training dataset to get the interacton and idxs
# We define a trainloader to split into batches
# We use the prepare input function here define a collator to get the target input token id, and target output token id. (We do next token prediction, so concenate input and output and fit the model on that)
# We pass in the dataset and collator, the llm to the trainer, and train the model

class LLMRanker(SequentialRecommender):
    def __init__(self, config, dataset, model, tokenizer):
        super().__init__(config, dataset)
        
        self.model = model.to(config['device'])
        self.model.eval()
        self.tokenizer = tokenizer
        self.llm_batch_size = config['llm_eval_batch_size']

        self.config = config
        self.max_tokens = config['max_tokens']
        self.api_model_name = config['api_name']
        openai.api_key = config['api_key']
        openai.api_base = config['api_base']
        self.api_batch = config['api_batch']
        self.async_dispatch = config['async_dispatch']
        self.temperature = config['temperature']

        self.max_his_len = config['max_his_len']
        self.recall_budget = config['recall_budget']
        self.boots = config['boots']
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.id_token = dataset.field2id_token['item_id']
        self.item_text = self.load_text()
        self.logger.info(f'Avg. t = {np.mean([len(_) for _ in self.item_text])}')

        self.fake_fn = torch.nn.Linear(1, 1)

    def load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        if self.dataset_name in ['ml-1m', 'ml-1m-full']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, release_year, genre = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name in ['Games', 'Games-6k']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, title = line.strip().split('\t')
                    token_text[item_id] = title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        else:
            raise NotImplementedError()
        
    
    def prepare_inputs(self, interaction, idxs):
        
        # origin_batch_size = idxs.shape[0]
        if self.boots:
            """ 
            bootstrapping is adopted to alleviate position bias
            `fix_enc` is invalid in this case"""
            idxs = np.tile(idxs, [self.boots, 1])
            np.random.shuffle(idxs.T)
        batch_size = idxs.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]
        prompt_list = []
        for i in tqdm(range(batch_size)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(interaction, idxs, i)

            prompt = self.construct_prompt(self.dataset_name, user_his_text, candidate_text_order)
            # prompt_list.append([{'role': 'user', 'content': prompt}])
            
            prompt_list.append(prompt)
        
        return prompt_list
    
    
    def predict_on_subsets(self, interaction, idxs):
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return:
        """
        
        # print('interaction', interaction)
        # print('idx', idxs)
        
        origin_batch_size = idxs.shape[0]
        if self.boots:
            """ 
            bootstrapping is adopted to alleviate position bias
            `fix_enc` is invalid in this case"""
            idxs = np.tile(idxs, [self.boots, 1])
            np.random.shuffle(idxs.T)
        batch_size = idxs.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]
        
        batch_size = 1 # Test on only 5 examples
        pos_items = pos_items[:1]
        
        print('############ Batch Size:', batch_size)
        
        ## All the data are loaded here, split into smaller batch_size to input into the model

        prompt_list = []
        for i in tqdm(range(batch_size)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(interaction, idxs, i)

            prompt = self.construct_prompt(self.dataset_name, user_his_text, candidate_text_order)
            # prompt_list.append([{'role': 'user', 'content': prompt}])
            
            prompt_list.append(prompt)
        
        # print('prompt_list', len(prompt_list), prompt_list)
        
        # if 'llama' in self.api_model_name:
        #     openai_responses = self.dispatch_replicate_api_requests(prompt_list, batch_size)
        # else:
        #     openai_responses = self.dispatch_openai_api_requests(prompt_list, batch_size)
        
        responses = self.get_batch_outputs(prompt_list, self.llm_batch_size)    
            
        save_responses_to_file(responses, f"responses_{self.api_model_name}.json")

        scores = torch.full((idxs.shape[0], self.n_items), -10000.)
        
        for i, response in enumerate(tqdm(responses)):

            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(interaction, idxs, i)

            response_list = response.split('\n')
            
            # print('response_list', len(response_list), response_list)
            
            # example:
            # ['1. Just the Ticket', '2. Victor/Victoria', "3. Love's Labour's Lost", '4. Sweet Nothing', '5. Passion Fish', "6. There's Something About Mary", '7. The Masque of the Red Death', '8. Friday the 13th Part VIII: Jason Takes Manhattan', '9. Fantasia', '10. Eyes Without a Face', '11. One False Move', '12. Double Indemnity', '13. Muppets From Space', '14. Instinct', '15. Heartburn', '16. The Cowboy Way', '17. Went to Coney Island on a Mission From God... Be Back by Five', '18. Beefcake', '19. Kronos', '20. Single Girl, A (La Fille Seule)']
            
            # self.logger.info(prompt_list[i])
            # self.logger.info(response)
            # self.logger.info(f'Here are candidates: {candidate_text}')
            # self.logger.info(f'Here are answer: {response_list}')
            
            if self.dataset_name in ['ml-1m', 'ml-1m-full']:
                rec_item_idx_list = self.parsing_output_text(scores, i, response_list, idxs, candidate_text)
            elif self.dataset_name in ['Games', 'Games-6k']:
                # rec_item_idx_list = self.parsing_output_indices(scores, i, response_list, idxs, candidate_text)
                rec_item_idx_list = self.parsing_output_text(scores, i, response_list, idxs, candidate_text)
            else:
                raise NotImplementedError()

            if int(pos_items[i % origin_batch_size]) in candidate_idx:
                target_text = candidate_text[candidate_idx.index(int(pos_items[i % origin_batch_size]))]
                try:
                    ground_truth_pr = rec_item_idx_list.index(target_text)
                    self.logger.info(f'Ground-truth [{target_text}]: Ranks {ground_truth_pr}')
                    # retry_flag = -1
                except:
                    # if 'llama' in self.api_model_name:
                    #     retry_flag = -1
                    #     pass
                    # else:
                    self.logger.info(f'Fail to find ground-truth items.')
                    print(target_text)
                    print(rec_item_idx_list)
                    # print(f'Remaining {retry_flag} times to retry.')
                    #     retry_flag -= 1
                    #     while True:
                    #         try:
                    #             openai_response = dispatch_single_openai_requests(prompt_list[i], self.api_model_name, self.temperature)
                    #             break
                    #         except Exception as e:
                    #             print(f'Error {e}, retry at {time.ctime()}', flush=True)
                    #             time.sleep(20)
                    
                    
            # else:
            #     retry_flag = -1

        if self.boots:
            scores = scores.view(self.boots,-1,scores.size(-1))
            scores = scores.sum(0)
        
        return scores

    def get_batch_inputs(self, interaction, idxs, i):
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        origin_batch_size = user_his.size(0)
        real_his_len = min(self.max_his_len, user_his_len[i % origin_batch_size].item())
        user_his_text = [str(j) + '. ' + self.item_text[user_his[i % origin_batch_size, user_his_len[i % origin_batch_size].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]
        candidate_text = [self.item_text[idxs[i,j]]
                for j in range(idxs.shape[1])]
        candidate_text_order = [str(j) + '. ' + self.item_text[idxs[i,j].item()]
                for j in range(idxs.shape[1])]
        candidate_idx = idxs[i].tolist()

        return user_his_text, candidate_text, candidate_text_order, candidate_idx

    def construct_prompt(self, dataset_name, user_his_text, candidate_text_order):
        if dataset_name in ['ml-1m', 'ml-1m-full']:
            prompt = f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif dataset_name in ['Games', 'Games-6k']:
            prompt = f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
                    # f"Please only output the order numbers after ranking. Split these order numbers with line break."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt

    # def dispatch_openai_api_requests(self, prompt_list, batch_size):
    #     openai_responses = []
    #     self.logger.info('Launch OpenAI APIs')
    #     if self.async_dispatch:
    #         self.logger.info('Asynchronous dispatching OpenAI API requests.')
    #         for i in tqdm(range(0, batch_size, self.api_batch)):
    #             while True:
    #                 try:
    #                     openai_responses += asyncio.run(
    #                         dispatch_openai_requests(prompt_list[i:i+self.api_batch], self.api_model_name, self.temperature)
    #                     )
    #                     break
    #                 except Exception as e:
    #                     print(f'Error {e}, retry batch {i // self.api_batch} at {time.ctime()}', flush=True)
    #                     time.sleep(20)
    #     else:
    #         self.logger.info('Dispatching OpenAI API requests one by one.')
    #         for message in tqdm(prompt_list):
    #             openai_responses.append(dispatch_single_openai_requests(message, self.api_model_name, self.temperature))
    #     self.logger.info('Received OpenAI Responses')
    #     return openai_responses
    
    # def dispatch_replicate_api_requests(self, prompt_list, batch_size):
    #     responses = []
    #     self.logger.info('Launch Replicate APIs')
    #     suffix = {
    #         'llama-2-7b-chat': '4b0970478e6123a0437561282904683f32a9ed0307205dc5db2b5609d6a2ceff',
    #         'llama-2-70b-chat': '2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1'
    #     }[self.api_model_name]
    #     for message in tqdm(prompt_list):
    #         while True:
    #             try:
    #                 output = replicate.run(
    #                     f"meta/{self.api_model_name}:{suffix}",
    #                     input={"prompt": f"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{message[0]['content']}[/INST]"}
    #                 )
    #                 break
    #             except Exception as e:
    #                 print(f'Error {e}, retry at {time.ctime()}', flush=True)
    #                 time.sleep(20)

    #         responses.append(''.join([_ for _ in output]))
    #     return responses
    
    def get_batch_outputs(self, prompts, batch_size):
        """
        Process prompts in batches and return the decoded responses.
        
        Args:
            prompts (list): List of prompts to process.
            batch_size (int): Size of each batch.
            
        Returns:
            list: Decoded responses for all prompts.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        all_responses = []  # To store the gathered responses

        # Split prompts into batches
        # Initialize the progress bar
        progress_bar = tqdm(range(0, len(prompts), batch_size), desc="Processing batches", ncols=100)

        # Split prompts into batches
        for i in progress_bar:
            batch_prompts = prompts[i:i + batch_size]  # Get the current batch
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors='pt',
                return_token_type_ids=False,
                padding=True,          # Pad to the longest sequence in the batch
                truncation=True,       # Truncate inputs that exceed the model's max length
                max_length=512         # Optional: Adjust max_length as needed
            ).to(self.config['device'])
            
            # Generate responses for the current batch
            responses = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Decode the responses and add to the result list
            batch_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            all_responses.extend(batch_responses)

        return all_responses
    
    # def get_batch_outputs(self, prompts, batch_size):
    #     # Initialize vLLM model (make sure this is done once; move it outside if reused)
    #     llm = LLM(self.api_model_name)  # Replace with the model's path or name

    #     # Define sampling parameters
    #     sampling_params = SamplingParams(
    #         temperature=self.temperature,  # Sampling temperature
    #         max_tokens=self.max_tokens,    # Maximum tokens to generate
    #         top_p=0.95,                    # Top-p (nucleus) sampling
    #         top_k=None                     # Optionally use top-k sampling
    #     )

    #     # Generate text for the batch of prompts
    #     results = llm.generate(prompts, sampling_params=sampling_params)

    #     # Extract and collect generated outputs
    #     response_decoded = [result.outputs[0].text for result in results]

        # return response_decoded

    def parsing_output_text(self, scores, i, response_list, idxs, candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue
            if item_detail.endswith('candidate movies:'):
                continue
            pr = item_detail.find('. ')
            if item_detail[:pr].isdigit():
                item_name = item_detail[pr + 2:]
            else:
                item_name = item_detail

            matched_name = None
            for candidate_text_single in candidate_text:
                clean_candidate_text_single = html.unescape(candidate_text_single.strip())
                if (clean_candidate_text_single in item_name) or (item_name in clean_candidate_text_single) or (pylcs.lcs_sequence_length(item_name, clean_candidate_text_single) > 0.9 * len(clean_candidate_text_single)):
                    if candidate_text_single in rec_item_idx_list:
                        break
                    rec_item_idx_list.append(candidate_text_single)
                    matched_name = candidate_text_single
                    break
            if matched_name is None:
                continue

            candidate_pr = candidate_text.index(matched_name)
            scores[i, idxs[i, candidate_pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
        return rec_item_idx_list

    def parsing_output_indices(self, scores, i, response_list, idxs, candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue

            if not item_detail.isdigit():
                continue

            pr = int(item_detail)
            if pr >= self.recall_budget:
                continue
            matched_name = candidate_text[pr]
            if matched_name in rec_item_idx_list:
                continue
            rec_item_idx_list.append(matched_name)
            scores[i, idxs[i, pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
            if len(rec_item_idx_list) >= self.recall_budget:
                break

        return rec_item_idx_list
