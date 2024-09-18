from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
import sys
from Executor import Executor
from utils import *
from bw_utils import *
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import yaml
import json

import pandas as pd
#from tarski.io import PDDLReader

# def get_problem(instance, domain):
#     reader = PDDLReader(raise_on_error=True)
#     reader.parse_domain(domain)
#     return reader.parse_instance(instance)

class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        tokenizer,
        train_size=0.2,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        #with open('data/blocksworld/bw_config.yaml', 'r') as file:
            #self.data = yaml.safe_load(file)
        #self.prompts = None #json.load(open("data/blocksworld/my_mcts_prompts_update.json", 'r'))
        #with open('data/blocksworld/bw_config.yaml', 'r') as file:
            #self.config = yaml.safe_load(file)
        self.config = None #todo  !!!!
        self.domain_pddl = None #
        self.tokenizer = tokenizer
        self.args = args
        self.train_data = None
        self.val_data = None

    def convert_cube_to_string(self, states):
    # Create a formatted string for each side of the cube
        cube_str = ""
    
        # Upper (4 elements from state_1 to state_4)
        cube_str += "Upper:\n"
        cube_str += f"{states[0]} {states[1]}\n"
        cube_str += f"{states[2]} {states[3]}\n"
        
        # Front (4 elements from state_5 to state_8)
        cube_str += "Front:\n"
        cube_str += f"{states[4]} {states[5]}\n"
        cube_str += f"{states[6]} {states[7]}\n"
        
        # Down (4 elements from state_9 to state_12)
        cube_str += "Down:\n"
        cube_str += f"{states[8]} {states[9]}\n"
        cube_str += f"{states[10]} {states[11]}\n"
        
        # Left (4 elements from state_13 to state_16)
        cube_str += "Left:\n"
        cube_str += f"{states[12]} {states[13]}\n"
        cube_str += f"{states[14]} {states[15]}\n"
        
        # Right (4 elements from state_17 to state_20)
        cube_str += "Right:\n"
        cube_str += f"{states[16]} {states[17]}\n"
        cube_str += f"{states[18]} {states[19]}\n"
        
        # Back (4 elements from state_21 to state_24)
        cube_str += "Back:\n"
        cube_str += f"{states[20]} {states[21]}\n"
        cube_str += f"{states[22]} {states[23]}\n"
        
        return cube_str

    def setup_helper(self, data):
        # just a helper function, (static)
        # Extract features and target variables from the CSV data
        states = data.iloc[:, 1:25].values  # states are in columns 1 to 24
        moves = data['moves'].values        # moves is the 'moves' column
        steps_to_win = data['minmumStep_to_win'].values  # steps to win column

        # Tokenize the moves using the provided tokenizer (assuming moves are in text format)
        tokenized_moves = [self.tokenizer(move)['input_ids'] for move in moves]


        # Combine states and tokenized moves into the dataset (you can adjust this as needed)
        #all_data = [(state, tokenized_move, step) for state, tokenized_move, step in zip(states, tokenized_moves, steps_to_win)]
        all_data = self.convert_cube_to_string(states)
        return all_data #,states, moves, steps_to_win

    def setup(self, stage):
        train_data = pd.read_csv('data/cube/cube_train.csv')
        test_data = pd.read_csv('data/cube/cube_test.csv')

        # Optionally limit the number of prompts loaded
        if self.hparams.limit_prompts is not None:
            train_data = train_data[: self.hparams.limit_prompts]
            test_data = test_data[: self.hparams.limit_prompts]

        all_train_data= self.setup_helper(train_data)
        all_test_data= self.setup_helper(test_data)



        self.train_data = PromptDataPipe(all_train_data)
        self.val_data = PromptDataPipe(all_train_data)
        self.test_data = PromptDataPipe(all_test_data)


    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=1)

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
