import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from recbole.data.interaction import Interaction

class LLMRankerDataset(Dataset):
    def __init__(self, config, dataset):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.selected_user_suffix = config['selected_user_suffix']
        self.data_path = config['data_path']
        self.recall_budget = config['recall_budget']
        self.user_token2id = dataset.field2token_id['user_id']
        self.item_token2id = dataset.field2token_id['item_id']
        self.selected_uids, self.sampled_items = self.load_selected_users(config, dataset)
        self.item_text = self.load_text(config, dataset)
        self.boots = config['boots']
        self.max_his_len = config['max_his_len']
        self.interactions = None
        self.candidates = None

    def load_selected_users(self, config, dataset):
        selected_users = []
        sampled_items = []
        selected_user_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.selected_user_suffix}')
        user_token2id = dataset.field2token_id['user_id']
        item_token2id = dataset.field2token_id['item_id']
        with open(selected_user_file, 'r', encoding='utf-8') as file:
            for line in file:
                uid, iid_list = line.strip().split('\t')
                selected_users.append(uid)
                sampled_items.append([item_token2id[_] if (_ in item_token2id) else 0 for _ in iid_list.split(' ')])
        selected_uids = list([user_token2id[_] for _ in selected_users])
        return selected_uids, sampled_items

    def load_text(self, config, dataset):
        token_text = {}
        item_text = ['[PAD]']
        feat_path = os.path.join(config['data_path'], f'{dataset.dataset_name}.item')
        if dataset.dataset_name in ['ml-1m', 'ml-1m-full']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, _, _ = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for token in dataset.field2id_token['item_id']:
                if token == '[PAD]': continue
                raw_text = token_text.get(token, '')
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
        elif dataset.dataset_name in ['Games', 'Games-6k']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, title = line.strip().split('\t')
                    token_text[item_id] = title
            for token in dataset.field2id_token['item_id']:
                if token == '[PAD]': continue
                item_text.append(token_text.get(token, ''))
        else:
            raise NotImplementedError()
        return item_text

    def build(self, data):
        if self.config["eval_type"] == "RANKING":
            self.tot_item_num = data._dataset.item_num

        iter_data = (
            tqdm(
                data,
                total=len(data),
                ncols=100,
                desc="Evaluate",
            )
        )
        unsorted_selected_interactions = []
        unsorted_selected_pos_i = []
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            for i in range(len(interaction)):
                if interaction['user_id'][i].item() in self.selected_uids:
                    pr = self.selected_uids.index(interaction['user_id'][i].item())
                    unsorted_selected_interactions.append((interaction[i], pr))
                    unsorted_selected_pos_i.append((positive_i[i], pr))
        unsorted_selected_interactions.sort(key=lambda t: t[1])
        unsorted_selected_pos_i.sort(key=lambda t: t[1])
        selected_interactions = [_[0] for _ in unsorted_selected_interactions]
        selected_pos_i = [_[0] for _ in unsorted_selected_pos_i]
        new_inter = {
            col: torch.stack([inter[col] for inter in selected_interactions]) for col in selected_interactions[0].columns
        }
        selected_interactions = Interaction(new_inter)
        selected_pos_i = torch.stack(selected_pos_i)
        selected_pos_u = torch.arange(selected_pos_i.shape[0])

        if self.config['has_gt']:
            print('Has ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            for i in range(idxs.shape[0]):
                if selected_pos_i[i] in idxs[i]:
                    pr = idxs[i].numpy().tolist().index(selected_pos_i[i].item())
                    idxs[i][pr:-1] = torch.clone(idxs[i][pr+1:])

            idxs = idxs[:,:self.recall_budget - 1]
            if self.config['fix_pos'] == -1 or self.config['fix_pos'] == self.recall_budget - 1:
                idxs = torch.cat([idxs, selected_pos_i.unsqueeze(-1)], dim=-1).numpy()
            elif self.config['fix_pos'] == 0:
                idxs = torch.cat([selected_pos_i.unsqueeze(-1), idxs], dim=-1).numpy()
            else:
                idxs_a, idxs_b = torch.split(idxs, (self.config['fix_pos'], self.recall_budget - 1 - self.config['fix_pos']), dim=-1)
                idxs = torch.cat([idxs_a, selected_pos_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
        else:
            print('Does not have ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            idxs = idxs[:,:self.recall_budget]
            idxs = idxs.numpy()

        if self.config['fix_pos'] == -1:
            print('Shuffle ground truth')
            for i in range(idxs.shape[0]):
                np.random.shuffle(idxs[i])
        
        # Keep track of ground truth, for later use in constructing grouth truth ranking.

        self.interactions = selected_interactions
        self.candidates = idxs

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        # interaction = self.interactions[idx]
        # candidates = self.candidates[idx]
        user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(self.interactions, self.candidates, idx)
        prompt = self.construct_prompt(self.dataset.dataset_name, user_his_text, candidate_text_order)
        return prompt

    def get_batch_inputs(self, interaction, idxs, i):
        user_his = interaction['item_id_list']
        user_his_len = interaction['item_length']
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
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt

'''
# Example Usage:

movie_dataset = MovieRecommendationDataset(config, dataset)
movie_dataset.build(test_data)

dataloader = DataLoader(movie_dataset, batch_size=8)


'''

class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
        print(loaded_feat.shape)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        item2row_path = osp.join(self.config['data_path'], f'{self.dataset_name}_item_dataset2row.npy')
        item2row = np.load(item2row_path,allow_pickle=True).item()
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[item2row[int(token)]]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding


class VQRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = self.load_index()

    def load_index(self):
        import faiss
        if self.config['index_pretrain_dataset'] is not None:
            index_dataset = self.config['index_pretrain_dataset']
        else:
            index_dataset = self.dataset_name
        index_path = os.path.join(
            self.config['index_path'],
            index_dataset,
            f'{index_dataset}.{self.index_suffix}'
        )
        self.logger.info(f'Index path: {index_path}')
        uni_index = faiss.read_index(index_path)
        old_pq_codes, _, _, _ = self.parse_faiss_index(uni_index)
        old_code_num = old_pq_codes.shape[0]

        self.plm_suffix = self.config['plm_suffix']
        self.plm_size = self.config['plm_size']
        feat_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        uni_index.add(loaded_feat)
        all_pq_codes, centroid_embeds, coarse_embeds, opq_transform = self.parse_faiss_index(uni_index)
        pq_codes = all_pq_codes[old_code_num:]
        assert self.code_dim == pq_codes.shape[1], pq_codes.shape
        # assert self.item_num == 1 + pq_codes.shape[0], pq_codes.shape

        # uint8 -> int32 to reserve 0 padding
        pq_codes = pq_codes.astype(np.int32)
        # 0 for padding
        pq_codes = pq_codes + 1
        # flatten pq codes
        base_id = 0
        for i in range(self.code_dim):
            pq_codes[:, i] += base_id
            base_id += self.code_cap + 1

        mapped_codes = np.zeros((self.item_num, self.code_dim), dtype=np.int32)
        item2row_path = osp.join(self.config['data_path'], f'{self.dataset_name}_item_dataset2row.npy')
        item2row = np.load(item2row_path, allow_pickle=True).item()
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_codes[i] = pq_codes[item2row[int(token)]]
            
        self.plm_embedding = torch.FloatTensor(loaded_feat)
        return torch.LongTensor(mapped_codes)

    @staticmethod
    def parse_faiss_index(pq_index):
        import faiss
        vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
        assert isinstance(vt, faiss.LinearTransform)
        opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

        ivf_index = faiss.downcast_index(pq_index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
        centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)

        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = faiss.rev_swig_ptr(coarse_quantizer.get_xb(), ivf_index.pq.M * ivf_index.pq.dsub)
        coarse_embeds = coarse_embeds.reshape(-1)

        return pq_codes, centroid_embeds, coarse_embeds, opq_transform
