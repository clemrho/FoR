import random
import sys
from functools import partial
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers.trainer_pt_utils import LabelSmoother
from util import lora_to_base, base_to_lora
from game24_utils import *
import yaml
import json
import torch.nn.functional as F
from prompts.game24 import *
from util import *
import sympy
import prompter_cube
import prompt_cube_operation_util
import prompter_cube as promp_toolbox

def extract_last_state(content): #???
    match = re.findall(r'\[STATE \w+\]', content)
    if match:
        last_state_index = content.rfind(match[-1])
        if last_state_index != -1:
            # 提取最后一个 [STATE *] 到 \n 之间的内容
            last_state_content = content[last_state_index + len(match[-1]):]
            end_index = last_state_content.find('\n')
            if end_index != -1:
                return last_state_content[:end_index].strip()

class PktCubeTask(LightningModule):
    pass
    def __init__(
        self,
        args,
        model,
        logZ,
        tokenizer,
        replay_buffer,
        train_data=None,
        val_data=None,
    ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.logZ = logZ
        self.model = model
        self.tokenizer = tokenizer
        self.reward = None
        self.train_data = train_data
        self.val_data = val_data
        self.replay_buffer = replay_buffer
        self.value_cache = {}
        self.n_samples = 10
        self.test_set = set()
        self.lr = args.lr
        self.logZ_lr = args.logZ_lr
        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)
        self.ignore_token_id = LabelSmoother.ignore_index 

    def get_ll_batch(self,inputs,labels,attention_masks):
        with torch.no_grad():
            lora_to_base(self.model)
            outputs = self.model(inputs,attention_mask = attention_masks, labels=labels)
            loss = outputs.loss
            base_to_lora(self.model)
            return torch.exp(-loss)**(1/0.7)       

    def get_ll(self,query,ys):
            # 准备模型
            lora_to_base(self.model)
            ignor_token_ids = LabelSmoother.ignore_index
            input_prompt = cot_prompt_wrap(query,ys)
            inputs = self.tokenizer(input_prompt,return_tensors="pt").to(self.device)
            labels = inputs["input_ids"].clone()
            generate_text = "Input: "  + query + '\n' + "Steps: \n" + ys
            labels[:,:len(inputs["input_ids"][0])-len(self.tokenizer(generate_text)["input_ids"])] = ignor_token_ids
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            base_to_lora(self.model)
            return torch.exp(-loss)**(1/0.7), inputs,labels

    def batch_preprocess(self, preprocessed_samples):
    # 找出最大的长度
        max_length = max(sample['input_ids'].shape[-1] for sample in preprocessed_samples)

        # 初始化堆叠后的数据
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for sample in preprocessed_samples:
            # 计算每个样本需要填充的长度
            padding_length = max_length - sample['input_ids'].shape[-1]
            
            padded_input_ids = torch.cat([sample['input_ids'], torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)] ,dim=-1)
            padded_attention_mask = torch.cat([sample['attention_mask'], torch.zeros(padding_length, dtype=torch.long).unsqueeze(0)], dim=-1)
            
            # 对labels进行填充
            padded_labels = torch.cat([sample['labels'], torch.full((padding_length,), self.ignore_token_id, dtype=torch.long).unsqueeze(0)], dim=-1)
            
            # 将处理后的数据加入到列表中
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)
        
        # 将列表转换为tensor
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
        batch_labels = torch.cat(batch_labels, dim=0)
        
        # 返回处理后的批次数据
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
    
    def query_LM(self,prompt,eos_token_id,do_sample = True, temperature = 0.7):
        temperature = temperature if do_sample else 0
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').cuda()
        attention_mask = torch.ones_like(input_ids)
        results = self.model.generate(input_ids, do_sample=True, max_new_tokens=20,top_k=10, attention_mask = attention_mask,use_cache=False)
        results = results[0][len(input_ids[0]):]
        results = self.tokenizer.decode(results, skip_special_tokens=False)
        lines = results.splitlines()
        first_line_after_prompt = lines[0] if lines else None
        return results #first_line_after_prompt # return 
        
    def generate_proposals(self,x, y,o_cube_status,do_val=False):
        """Input:

        cube_status: The current cube status (e.g., a string like 1,1,1,1; 2,2,2,2; ...).
        Output:

        A flag indicating if the spin was successful (True/False).
        A string ans_str describing the spin in a readable format.
        The new cube state (num_list), which is the cube state after performing the spin.
        Process:

        The function defines a list of legal cube operations (U, R, F, etc.).
        It randomly selects an operation from this list 
        The selected operation is applied to the original cube status by calling doAlgStr(orig_color_list, operation), which modifies the cube based on the operation.
        The result is formatted into a readable string (readable_res) and returned along with the updated cube state.
        """
    #     # print(nums)
    #     # return generate_op(nums)
        propose_prompt = promp_toolbox.cot_prompt_wrap(x, y) 
        proposals = self.query_LM(propose_prompt, eos_token_id=self.tokenizer.eos_token_id)
        # check whether feasible 
        #flag, proposals, numss = calculate_and_complete_expression(proposals,nums)#
        flag, readable_proposals, a_cube_status = prompt_cube_operation_util.generate_spin(proposals, o_cube_status)   
        if do_val and not flag: return None, None
        calc = 0
        while(calc < 2):
            if flag:
                return readable_proposals, a_cube_status             
            else: #generate again
                flag, readable_proposals, a_cube_status= prompt_cube_operation_util.generate_spin(proposals,o_cube_status)#
                calc+= 1
                if flag: 
                    return readable_proposals, a_cube_status
        return prompt_cube_operation_util.conduct_spin(o_cube_status)    #if all failed, do manually.   

    def get_value(self, x, y, n_evaluate_sample, cache_value=True):
        value_prompt = value_prompt_wrap(x, y)
        # print(value_prompt)
        if cache_value and value_prompt in self.value_cache:
            return self.value_cache[value_prompt]
        value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
        # print(value_outputs)
        # input()
        value = value_outputs_unwrap(x, y, value_outputs)
        if cache_value:
            self.value_cache[value_prompt] = value
        return value
    
    # def test_output(self, query, output: str):
    #     expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
    #     numbers = re.findall(r'\d+', expression)
    #     problem_numbers = re.findall(r'\d+', query)
    #     if sorted(numbers) != sorted(problem_numbers):
    #         return {'r': 0}
    #     try:
    #         # print(sympy.simplify(expression))
    #         return {'r': int(sympy.simplify(expression) == 24)}
    #     except Exception as e:
    #         # print(e)
    #         return {'r': 0} #  this is also useless.
        
    

    def get_2x2_cube_thoughts(self, query, use_gpt_val = False, do_val = False):
    #"""
    #   Input (x)   : the origin cube model
    #     Output (y)  : a trajectory of at most 11 steps to reach final state
    #     Reward (r)  : TBD negative..
    #
     # 1. generated_text
        ys = '' #current output
        sample = None
        ll_reward = None
        infos = []
        values = []
         # print(query)
        state_nums = query.split()

        for step in range(11):
            new_ys, state_nums= self.generate_proposals(query,ys,state_nums,do_val) # ask model 
            if new_ys is None and do_val:
                return "FAIL", None, None, None, [0]
            if use_gpt_value:
                values.append(self.get_value(query, new_ys, 1, True)) # ask gpt-4
            infos.append({'step': step, 'x': query, 'ys': ys, 'new_ys': new_ys, 'values': values})
            ys = ys  + new_ys + '\n'
        output = cot_prompt_wrap(query,ys)
        ll_reward, inputs, labels = self.get_ll(query,ys)
        attention_mask = torch.ones_like(inputs["input_ids"]).to('cpu')
        sample = dict(
            input_ids = inputs["input_ids"].to('cpu'),
            labels = labels.to('cpu'),
            attention_mask = attention_mask
        )
        if use_gpt_value:
            reward = 1
        else:
            reward = 0.0001;
        output = "Input: " + query + '\n' + "Steps: \n" + ys

        
        return  output, sample, reward,ll_reward, state_nums           


    
    
    def forward_prob(self, input_ids, targets, attention_mask):
        base_to_lora(self.model)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        N, L, C = logits.shape  

        loss_per_token = F.cross_entropy(shift_logits.view(-1, C), shift_labels.view(-1), ignore_index=self.ignore_token_id, reduction='none')

        loss_per_sample = loss_per_token.view(N, L-1)

        
        loss_per_sample = loss_per_sample * attention_mask[..., 1:]

        
        loss_per_sample = loss_per_sample.sum(dim=1)
        return loss_per_sample
    
    def training_step(self,batch,batch_idx):
        LOG_PF, LOG_R = [], []


        batch = tuple(t.to(self.device) for t in batch) 
        input_ids, labels, attention_mask = batch
        if self.args.do_sft:
            base_to_lora(self.model)
            outputs = self.model(input_ids,attention_mask = attention_mask, labels=labels)
            loss = outputs.loss
            self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
            return loss
        ll_rewards = self.get_ll_batch(input_ids,attention_masks=attention_mask,labels=labels)
        log_reward_list = torch.log(rewards + ll_rewards)
        LOG_PF = self.forward_prob(input_ids, labels, attention_mask)
        LOG_R.extend(log_reward_list)
        LOG_R = torch.stack(LOG_R)
    
        loss = tb_loss(
            log_pf=LOG_PF,
            log_r=LOG_R,
            logz=self.logZ
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )

        return loss

    def validation_step(self,problem,batch_idx):
        print("===Validation===")
        test_nums = 5
        success = False
        self.model.eval()
        suc_num = 0
        for i in range(test_nums):
            # output,sample,reward,ll_reward,sn = self.get_24_plans(problem[0],use_gpt_value=False)
            # if sn[0]=='24':
            #     suc_num += 1
            # if not success: success = (sn[0]=='24')
            # print(output)
            output, sample, reward, ll_reward, sn = self.get_2x2_cube_thoughts(problem[0],use_gpt_value=False)
            if prompt_cube_operation_util.isSolved(sn):
                suc_num += 1
            if not success:
                success = (prompt_cube_operation_util.isSolved(sn))
            print(output)
            # if sn is a valid answer: ...

        print("SUC_NUM: ",suc_num)
        self.log(
            "val/success",
            success,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
    
    def test_step(self,problem,batch_idx):
        print("===Test===")
        test_nums = self.args.test_sample_nums
        success = False
        self.model.eval()
        self.test_set = set()
        for i in range(test_nums):
            output,sample,reward,ll_reward,sn = self.get_24_plans(problem[0],use_gpt_value=False,do_val=True)
            if prompt_cube_operation_util.isSolved(sn):
                self.test_set.add(output)
            if not success: success = (prompt_cube_operation_util.isSolved(sn))
            print(output)
        print("TRAJ_NUM: ",len(self.test_set))
        self.log(
            "val/success",
            success,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "val/traj_num",
            len(self.test_set),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
    
    def configure_optimizers(self):
        
        if self.args.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            return bnb.optim.PagedAdamW8bit([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])
        else:
            return torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])
