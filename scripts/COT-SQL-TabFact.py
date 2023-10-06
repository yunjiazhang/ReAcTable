#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


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
from AutoReasoner import *
import dotenv

config = dotenv.dotenv_values(".env")
openai.api_type = 'azure'
openai.api_base = 'https://meta-prompter-az-openai.openai.azure.com'
openai.api_version = '2022-12-01'
openai.api_key = config['OPENAI_API_KEY_ms']

dataset = pd.read_csv('./dataset/Table-Fact-Checking/small_test.csv', sep=',')
# dataset = pd.read_csv('./dataset/Table-Fact-Checking/train_sample.csv', sep=',')
# dataset = pd.read_csv('./dataset/WikiTableQuestions/data/training.tsv', sep='\t')

ft=None


# In[3]:


# import fasttext
# ft = fasttext.load_model('/mnt/idm_automapping/cc.en.300.bin')


# In[10]:


NNDemo = False
max_demo = 7
template = 'original-sql'
# template = 'formatv1'
gpt_model = 'mp-aoi-codex'

def parallel_codex_func(i):
    max_retry = 3
    while max_retry>0:
        try:
            codex_prompter = CodexAnswerCOTExecutor_template(
                                              f'./prompt_template/{template}.json',
                                              # '/mnt/text2sql/dataset/Table-Fact-Checking/prompt_template/formatv1.json',
                                              dataset.iloc[i]['id'], 
                                              dataset.iloc[i]['utterance'], 
                                              './data/all_csv/' + dataset.iloc[i]['context'], 
                                              dataset.iloc[i]['targetValue'], 
                                              base_path='./dataset/Table-Fact-Checking/',
                                              demo_file=f'few-shot-demo/TabFact-formatv1.json',
                                             sep='#'
                                             )
            codex_prompter.model = gpt_model
            codex_prompter.max_demo = max_demo
            
            codex_prompter._gen_gpt_prompt()
            codex_prompter._get_gpt_prediction()
            log = codex_prompter._log_dict()
            break
        except Exception as e:
            log = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            max_retry -= 1
    return log

for program in [ 'sql', ]:
    n_threads = 1
    maxLimit = float('inf')
    # maxLimit = 10
    from joblib import Parallel, delayed
    logs = Parallel(n_jobs=n_threads, require='sharedmem')(delayed(parallel_codex_func)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0]))))
    json.dump(logs, open(f'./dataset/Table-Fact-Checking/results/CodexAnswerCOTExecutor_{template}_{program}_NNDemo={NNDemo}_results_test_small_limit{maxLimit}_model{gpt_model}.json', 'w'), indent=4)
    correct_cnt = 0
    for l in logs:
        if 'predicted_value' in l and l['target_value'] == l['predicted_value']:
            correct_cnt += 1
    print(f"Acc = {correct_cnt} / {len(logs)} = {correct_cnt / len(logs)}")


# In[9]:


logs[0]


# In[7]:


def parallel_codex_func(i):

    codex_prompter = CodexAnswerCOTExecutor_template(
                                      f'/mnt/text2sql/dataset/Table-Fact-Checking/prompt_template/{template}.json',
                                      # '/mnt/text2sql/dataset/Table-Fact-Checking/prompt_template/formatv1.json',
                                      dataset.iloc[i]['id'], 
                                      dataset.iloc[i]['utterance'], 
                                      './data/all_csv/' + dataset.iloc[i]['context'], 
                                      dataset.iloc[i]['targetValue'], 
                                      base_path='./dataset/Table-Fact-Checking/',
                                      demo_file=f'few-shot-demo/TabFact-formatv1.json',
                                     sep='#'
                                     )
    codex_prompter.model = gpt_model
    codex_prompter.max_demo = max_demo
    # codex_prompter._gen_gpt_prompt()
    codex_prompter._gen_gpt_prompt()
    codex_prompter._get_gpt_prediction()
    log = codex_prompter._log_dict()
    return log
parallel_codex_func(0)


# In[ ]:




