#!/usr/bin/env python
# coding: utf-8

# In[3]:


# %load_ext autoreload
# %autoreload 2


# In[1]:


import re
import pandas as pd
import openai
import os
import json
# from gpt3_sandbox.api.gpt import GPT
# from gpt3_sandbox.api.gpt import Example
from pandasql import sqldf
from tqdm import tqdm
import numpy as np
from GptPrompter import *
from GptCOTPrompter import *
from GptCOTPrompter_BeamSeach import *
from AutoReasoner import *
import dotenv

config = dotenv.dotenv_values(".env")
openai.api_type = 'azure'
openai.api_base = 'https://meta-prompter-az-openai.openai.azure.com'
openai.api_version = '2022-12-01'
openai.api_key = config['OPENAI_API_KEY_ms']

# print(openai.Model.list())
dataset = pd.read_csv('./dataset/WikiTableQuestions/data/pristine-unseen-tables.tsv', sep='\t')
# dataset = pd.read_csv('./dataset/WikiTableQuestions/data/pristine-unseen-tables-sample400.tsv', sep='\t')
# dataset = pd.read_csv('./dataset/WikiTableQuestions/data/training.tsv', sep='\t')
ft = None


# In[2]:


# import fasttext
# ft = fasttext.load_model('cc.en.300.bin')


# In[5]:


# import sys
# print(sys.executable)


# In[3]:


NNDemo = False
max_demo = 5
# template = 'original-sql-py'
template = 'original-sql-py-no-intermediate'
# template = 'formatv1-sql-py'

# gpt_model = 'text-davinci-003'
gpt_model = 'mp-aoi-codex'
# gpt_model = 'gpt-3.5-turbo'

def parallel_codex_func_formatv1(i):
    max_retry = 3
    while max_retry>0:
        try:
            codex_prompter = CodexAnswerCOTExecutor_LeverVote(
                                              f'prompt_template/{template}.json',
                                              dataset.iloc[i]['id'], 
                                              dataset.iloc[i]['utterance'], 
                                              dataset.iloc[i]['context'], 
                                              dataset.iloc[i]['targetValue'],  
                                              base_path='./dataset/WikiTableQuestions/',
                                              demo_file=f'few-shot-demo/WikiTQ-{program}.json',
                                             )
            codex_prompter.model = gpt_model
            codex_prompter.max_demo = max_demo
            
            # codex_prompter.demo_ids = [0, 1, 2, 3, 6, 8, 11]
            # codex_prompter._gen_gpt_prompt()
            
            codex_prompter._gen_gpt_prompt(NNDemo, ft)
            codex_prompter._get_gpt_prediction(5)
            log = codex_prompter._log_dict()
            break
        except Exception as e:
            log = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            if "model's maximum context length" in str(e):
                return log
            max_retry -= 1
    return log
    
for program in ['sql-py']:
    n_threads = 1
    maxLimit = float('inf')
    # maxLimit = 1
    from joblib import Parallel, delayed
    output_result_file = f'./dataset/WikiTableQuestions/results/CodexAnswerCOTExecutor_LeverVote_{template}_{program}_NNDemo={NNDemo}_results_pristine-unseen-tables_limit{maxLimit}_model{gpt_model}.json'
    logs = Parallel(n_jobs=n_threads, require='sharedmem')(delayed(parallel_codex_func_formatv1)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0]))))
    json.dump(logs, open(output_result_file, 'w'), indent=4)
    # evaluate: 
    os.system(f'cd ./dataset/WikiTableQuestions/ && python2 evaluator.py ./results/{output_result_file.split("/")[-1]} && cd ..')


# In[10]:


NNDemo = False
max_demo = 5
template = 'original-sql-py-no-intermediate'
program = 'sql-py'
# gpt_model = 'text-davinci-003'
gpt_model = 'mp-aoi-codex'

def func(i):
    codex_prompter = CodexAnswerCOTExecutor_LeverVote(
                                      f'prompt_template/{template}.json',
                                      dataset.iloc[i]['id'], 
                                      dataset.iloc[i]['utterance'], 
                                      dataset.iloc[i]['context'], 
                                      dataset.iloc[i]['targetValue'],  
                                      base_path='./dataset/WikiTableQuestions/',
                                      demo_file=f'few-shot-demo/WikiTQ-{program}.json',
                                     )
    codex_prompter.model = gpt_model
    codex_prompter.max_demo = max_demo
    codex_prompter.demo_ids = [0, 1, 2, 3, 6, 8, 11]


    # codex_prompter._gen_gpt_prompt()
    codex_prompter._gen_gpt_prompt(NNDemo, ft)
    codex_prompter._get_gpt_prediction(5)
    log = codex_prompter._log_dict()
    # print(vars(codex_prompter))
    # print(codex_prompter.frequency_penalty)
    return log
a = func(0)


# In[11]:


# print(a['prompt'])
# print(a)
a

