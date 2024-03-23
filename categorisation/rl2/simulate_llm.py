import numpy as np
import pandas as pd
import ipdb
import anthropic
from dotenv import load_dotenv
import argparse
load_dotenv() # load environment variables from .env
import time
import sys
SYS_PATH = '/u/ajagadish/ermi'
sys.path.append(f'{SYS_PATH}/categorisation/data')
from envs import ShepardsTask, NosofskysTask, SmithsTask, JohanssensTask

def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)


def call_claude(query, model="claude-2", temperature=0., max_tokens=1):
    # anthropic api call
    client = anthropic.Anthropic()
    response = client.completions.create(
            prompt = query,
            model=model,
            temperature=temperature,
            max_tokens_to_sample=max_tokens,
        ).completion.replace(' ', '')
    
    return response

def simulate_llm_on_shepard1961(mode='llm', model='claude-2', start_participant=0, from_env=False):
    datasets = ["data/human/badham2017deficits.csv"]
    all_prompts = []
    # categories = {'j': 'A', 'f': 'B'}

    for dataset in datasets:
        df = pd.read_csv(dataset)
        df['llm_category'], df['true_category'] = np.nan, np.nan # add new column to df to store the llm predicted category
        num_participants = 2 #df.participant.max() + 1
        num_tasks = df.task.max() + 1
        num_blocks = 6 if from_env else df.block.max() + 1
        start_block = 4 if from_env else 0
        # create a new dataframe that copies data from df[df['block'] == 3' to make to new df for block 5 and 6
        if from_env:
            # Create a copy of the dataframe where block == 3
            df_block3 = df[df['block'] == 3].copy()
            df_block4 = df_block3.copy().assign(block=4)
            df_block5 = df_block3.copy().assign(block=5)
            # set condition of df_block4 and df_block5 to 5 and 6 respectively
            df_block4['condition'] = 5
            df_block5['condition'] = 6
            # set choice and correct to nans
            df_block4['choice'] = np.nan
            df_block4['correct_choice'] = np.nan
            df_block5['choice'] = np.nan
            df_block5['correct_choice'] = np.nan
            df = pd.concat([df, df_block4, df_block5], ignore_index=True)
        
        for participant in range(start_participant, num_participants):
            df_participant = df[(df['participant'] == participant)]
            num_trials = df_participant.trial.max() + 1 # participant specific number of trials
            num_features = 3 #
            choice_options = randomized_choice_options(num_choices=2)
            categories = {'j': choice_options[0], 'f': choice_options[1]} 
 
            # instructions
            instructions =   'In this experiment, you will be shown examples of geometric objects. \n'\
            f'Each object has {num_features} different features: size, color, and shape. \n'\
            'Your job is to learn a rule based on the object features that allows you to tell whether each example \n'\
            f'belongs in the {str(choice_options[0])} or {str(choice_options[1])} category. As you are shown each \n'\
            'example, you will be asked to make a category judgment and then you \n'\
            'will receive feedback. At first you will have to guess, but you will \n'\
            'gain experience as you go along. Try your best to gain mastery of the \n'\
            f'{str(choice_options[0])} and {str(choice_options[1])} categories. \n\n'\

            for task in range(num_tasks):
                df_task = df_participant[(df_participant['task'] == task)]

                for block in range(start_block, num_blocks):

                    df_block = df_task[(df_task['block'] == block)] if from_env else df_task[(df_task['block'] == block)]
                    num_trials_block = df_block.trial.max() + 1 # block specific number of trials
                    block_instructions = instructions #+ f'In this block {block+1}, you will be shown {num_trials_block} examples of geometric objects. \n'
 
                    if from_env:
                        assert block >= 4, 'Block number should be greater than 4 for using environment'
                        env = ShepardsTask(task=block+1, return_prototype=True, batch_size=1, max_steps=96, shuffle_trials=True)
                        outputs = env.sample_batch()
                        packed_inputs, _, targets, _ = outputs
                        # make a dictionary: can take `Small`` or `Big`` if either 0 or 1; `Black`` or `White``, 2 dim to `Square` or `Triangle`, 
                        feature_dict = {0:{0: 'Small', 1: 'Big'}, 1:{0: 'Black', 1: 'White'}, 2:{0: 'Square', 1: 'Triangle'}}
                        # shuffle all key value pairs
                        np.random.shuffle(feature_dict[0])
                        np.random.shuffle(feature_dict[1])
                        np.random.shuffle(feature_dict[2])
                        

                    for t_idx, trial in enumerate(df_block.trial.values[:num_trials_block]):
                        df_trial = df_block[(df_block['trial'] == trial)]
                        
                        if from_env and (block>=4):
                            t = str(choice_options[targets[0][t_idx].numpy().astype(int)[0]])
                            # map the bar code input to the object feature in each dimension
                            barcode  = packed_inputs[0][t_idx][:num_features].numpy().astype(int)
                            object_name = ' '.join([feature_dict[i][barcode[i]] for i in range(len(barcode))])
                        else:
                            t = categories[df_trial.correct_choice.item()]
                            object_name = df_trial.object.item()
            

                        # anthropic prompt
                        Q_ = anthropic.HUMAN_PROMPT if model == 'claude-2' else 'Q:'
                        A_ = anthropic.AI_PROMPT if model == 'claude-2' else 'A:'
                        question = f'{Q_} What category would a ' + object_name + ' belong to? (Give the answer in the form \"Category <your answer>\").'\
                                f'{A_} Category'
                        query = block_instructions + question
                        # print(query)
                        count = 0
                        while count < 10:
                            try: 
                                time.sleep(3**count - 1)
                                llm_response = call_claude(query)
                                break
                            except Exception as e:
                                print(f'Error in anthropic {e}. Retrying...')
                                count += 1
                                continue
                    
                        # check if response is valid
                        assert llm_response in [str(choice_options[0]), str(choice_options[1])], 'Invalid response. Please try again.'


                        # add llm predicted category and true category to df
                        df.loc[(df['participant'] == participant) & (df['task'] == task) & (df['trial'] == trial) & (df['block'] == block) , 'llm_category'] = llm_response
                        df.loc[(df['participant'] == participant) & (df['task'] == task) & (df['trial'] == trial) & (df['block'] == block) , 'true_category'] = str(t)
                        # add objet name and  to df
                        df.loc[(df['participant'] == participant) & (df['task'] == task) & (df['trial'] == trial) & (df['block'] == block) , 'object'] = object_name

                        
                        block_instructions += '- In trial '+ str(t_idx+1) +', you saw ' + object_name + ' which belonged to category ' + str(t) + '.\n'

            # save df with llm predicted category and true category
            df.to_csv(dataset.replace('.csv', f'llm_choices{mode}_{start_participant}_shepard1961.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='llm', help='llm or human')
    parser.add_argument('--model', type=str, default='claude-2', help='claude-2 or claude-1')
    parser.add_argument('--dataset', type=str, default='badham2017', help='badham2017 or devraj2022')
    parser.add_argument('--start-participant', type=int, default=0, help='start participant number')
    parser.add_argument('--from-env', action='store_true', help='use environment or dataset')
    parser.add_argument('--num-runs', type=int, required=False,  default=1, help='number of runs')
    
    args = parser.parse_args()
    if args.dataset == 'badham2017':
        simulate_llm_on_shepard1961(mode=args.mode, model=args.model, start_participant=args.start_participant, from_env=args.from_env)

   