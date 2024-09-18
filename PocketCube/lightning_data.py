from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import random_split
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
from transformers.trainer_pt_utils import LabelSmoother
from game24_utils import *
import warnings
import sys
from util import *
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import yaml
import json
from tarski.io import PDDLReader
import pandas as pd
import torch
import random

def get_problem(instance, domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain)
    return reader.parse_instance(instance)


class InputExample():
    def __init__(self,input_ids, labels, attention_masks, reward):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_masks = attention_masks
        self.reward = reward


class PktCubeDataModule(LightningDataModule):
    def __init__(
            self,
            args,
            tokenizer,
            train_size = 0.8,
            device = "cuda",
            limit_prompts=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.train_data = None
        self.val_data = None
        self.train_size = train_size
        self.test_data = None
        self.device = device

    # def setup(self,stage=None):
    #     print(stage)
    #     if stage=="fit" or stage is None:
    #         print("Loading data")
    #         game24full = []
    #         with open(self.args.train_data,'r') as f:
    #             for line in f:
    #                 game24full.append(json.loads(line))
            
    #         features = self.convert_train_to_features(game24full)
    #         all_inputs_ids = torch.stack([f.input_ids for f in features])
    #         all_labels = torch.stack([f.labels for f in features])
    #         all_attention_masks = torch.stack([f.attention_masks for f in features])
    #         all_rewards = torch.Tensor([f.reward for f in features])
    #         train_data = TensorDataset(all_inputs_ids, all_labels, all_attention_masks, all_rewards)
    #         self.train_data = PromptDataPipe(train_data)
    #         #print("Loading validation data")
    #         tests_all = pd.read_csv('data/cube_test.csv')
    #         validation_indices = list(range(10, 20)) + list(range(-20, -10))
    #         val_data = tests_all.iloc[validation_indices].to_dict(orient='records')
    #         val_features = self.convert_train_to_features(val_data)
    #         self.val_data = PromptDataPipe(val_features)
            
    #     elif stage=='test':
    #         # tests_all = list(pd.read_csv('data/24.csv')['Puzzles'])
    #         # self.test_data = PromptDataPipe(tests_all[910:1010])
    #         print("Loading test data")
    #         tests_all = pd.read_csv('data/cube_test.csv')
    #         test_data = tests_all.iloc[910:1010].to_dict(orient='records')
    #         test_features = self.convert_train_to_features(test_data)
    #         self.test_data = PromptDataPipe(test_features)

    

    def creat_labels(self,inputs,generate_text,ignore_token_id):
        labels = inputs["input_ids"].clone()
        labels[:,:len(inputs["input_ids"][0])-len(self.tokenizer(generate_text)["input_ids"])] = ignore_token_id
        return labels


    # def convert_train_to_features(self, examples,max_length=1024):
    #     ignore_token_id = LabelSmoother.ignore_index 
    #     features = [] 
    #     i = 0
    #     query = ''
    #     for example in examples:
    #         if self.args.do_sft: 
    #             if example['reward'] != 100:
    #                 continue
    #             elif example['idx'] != query:
    #                 query = example['idx']
    #             else:
    #                 continue

        
    #         input_prompt = cot_prompt_wrap(example['input']) + example['generate_data']
    #         generate_text = "Steps: \n" + example['generate_data']
    #         inputs = self.tokenizer(input_prompt,return_tensors="pt")
    #         if max_length < len(inputs["input_ids"][0]):
    #             print("Input length is greater than max_length")
    #             print(inputs["input_ids"].shape[1])
    #             input()
    #         padding_length = max_length - inputs["input_ids"].shape[1]
    #         labels = self.creat_labels(inputs,generate_text,ignore_token_id)
    #         attention_mask = torch.ones_like(inputs["input_ids"])
    #         padded_input_ids = torch.cat([inputs['input_ids'], torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)] ,dim=-1)[0]
    #         padded_attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long).unsqueeze(0)], dim=-1)[0]
    #         padded_labels = torch.cat([labels, torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)], dim=-1)[0]
    #         features.append(InputExample(
    #             input_ids=padded_input_ids,
    #             labels=padded_labels,
    #             attention_masks=padded_attention_mask,
    #             reward=example['reward']
    #         ))
    #         if i < 5:
    #             print("***Example: ***", i)
    #             print("lenngth of input_ids:", len(inputs["input_ids"][0]))
    #             print("padded_input_tokens:", self.tokenizer.decode(padded_input_ids))
    #             i += 1
    #     return features
    def convert_train_to_features(self, examples,max_length=1024):
        ignore_token_id = LabelSmoother.ignore_index 
        features = [] 
        i = 0
        query = ''
        for example in examples:
            if self.args.do_sft: 
                if example['reward'] != 100:
                    continue
                elif example['idx'] != query:
                    query = example['idx']
                else:
                    continue

        
            input_prompt = cot_prompt_wrap(example[1:24]) + example['generate_data']
            generate_text = "Steps: \n" + example['generate_data']
            inputs = self.tokenizer(input_prompt,return_tensors="pt")
            if max_length < len(inputs["input_ids"][0]):
                print("Input length is greater than max_length")
                print(inputs["input_ids"].shape[1])
                input()
            padding_length = max_length - inputs["input_ids"].shape[1]
            labels = self.creat_labels(inputs,generate_text,ignore_token_id)
            attention_mask = torch.ones_like(inputs["input_ids"])
            padded_input_ids = torch.cat([inputs['input_ids'], torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)] ,dim=-1)[0]
            padded_attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long).unsqueeze(0)], dim=-1)[0]
            padded_labels = torch.cat([labels, torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)], dim=-1)[0]
            features.append(InputExample(
                input_ids=padded_input_ids,
                labels=padded_labels,
                attention_masks=padded_attention_mask,
                reward=example['reward']
            ))
            if i < 5:
                print("***Example: ***", i)
                print("lenngth of input_ids:", len(inputs["input_ids"][0]))
                print("padded_input_tokens:", self.tokenizer.decode(padded_input_ids))
                i += 1
        return features
    

    def setup(self, stage=None):
        print(stage)
        if stage == "fit" or stage is None:
            print("Loading data")
            data = pd.read_csv(self.args.train_data)
            # Convert DataFrame to list of dicts
            data_dicts = data.to_dict(orient='records')

            # Transform data format
            transformed_data = []
            for item in data_dicts:
                transformed = {
                    'states': [item[f'state_{i}'] for i in range(1, 25)],
                    'moves': item['moves'],
                    'minimum_step_to_win': item['minmumStep_to_win']
                }
                transformed_data.append(transformed)

            features = self.convert_train_to_features(transformed_data)
            all_input_ids = torch.stack([f['input_ids'] for f in features])
            all_labels = [f['labels'] for f in features]
            all_attention_masks = torch.stack([f['attention_masks'] for f in features])
            all_rewards = torch.Tensor([f['reward'] for f in features])

            train_data = TensorDataset(all_input_ids, all_attention_masks, all_rewards)
            self.train_data = PromptDataPipe(train_data)

    
    def train_dataloader(self):
        return DataLoader(self.train_data,batch_size=self.args.batch_size,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1) 
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1)


class PromptDataPipe(MapDataPipe):
    def __init__(self, problems) -> None:
        super().__init__()
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):

        return self.problems[index]