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
import itertools
import random
from natsort import natsort_keygen
import pylcs
from tokenizer import *
# from COT_py_func import *
import re

import dotenv
config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY_ms']


def get_utterance_embedding(utterance, fasttext_model):
    words = utterance.split(' ')
    return np.mean([fasttext_model[word.lower()] for word in words], axis=0)

def get_embedding_cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_NN_demo(utterance, all_demo_embeddings, fasttext_model, top_n=3):
    embedding = get_utterance_embedding(utterance, fasttext_model)
    all_dist = np.matmul(embedding, all_demo_embeddings) / np.linalg.norm(all_demo_embeddings, axis=0)
    ids = np.argsort(all_dist)[::-1][:top_n]
    return [i for i in ids.tolist()]

def table_formater(df, seperator='|', col_data_split='-', 
                   col_prefix='[HEAD]', row_prefix='[ROW]', 
                   permute_df=False, utterance=None, 
                   key_val_pairs=False, line_limit=float('inf')):
    cols = [c.replace('\n', ' ').replace(' ', '_').lower() for c in df.columns.tolist()]
    table_str = []
    if col_prefix != '':
        table_str.append(col_prefix + ': ' + seperator.join(cols))
    else:
        table_str.append(seperator.join(cols))
        
    if col_data_split != '':
        table_str.append(col_data_split * 3)
        
    too_long = False
    if df.shape[0] == 0:
        table_str.append("EMPTY TABLE")
    for i in range(df.shape[0]):
        if i >= line_limit-2 and i<df.shape[0] - 2:
            if not too_long:
                too_long = True
                table_str.append('...')
            continue
        if row_prefix != '':
            if key_val_pairs:
                table_str.append(row_prefix + f' {i+1}: ')
                table_str[-1] += seperator.join([ f'{c}:' + str(i).replace('nan', 'NULL') for c, i in zip(cols, df.iloc[i].tolist())])
            else:    
                table_str.append(row_prefix + f' {i+1}: ' + seperator.join([str(i).replace('nan', 'NULL').replace('\n', ' ') for i in df.iloc[i].tolist()]))
        else:
            if key_val_pairs:
                table_str.append(seperator.join([ f'{c}:' + str(i).replace('nan', 'NULL').replace('\n', ' ') for c, i in zip(cols, df.iloc[i].tolist())]))
            else:
                table_str.append(seperator.join([str(i).replace('nan', 'NULL').replace('\n', ' ') for i in df.iloc[i].tolist()]))
    return '\n'.join(table_str)

def shuffleDataFrame(df, n=10):
    all_possible_varients = {}
    new_df = df.copy()
    all_possible_varients['default'] = new_df
    
    cols = df.columns.tolist()
    indices = df.index.tolist()
    while n > 0:
        new_df = df.copy()
        random.shuffle(cols)
        random.shuffle(indices)
        new_df = new_df.iloc[indices][cols]
        all_possible_varients[f'rand shuffle {n}'] = new_df
        n -= 1
    return all_possible_varients

def permuteDataFrame(df, utterance=None, ft=None):
    all_possible_varients = {}
    cols = df.columns.tolist() 
    
    # original df
    new_df = df.copy()
    all_possible_varients['original df'] = new_df
    
    # sort by individual columns
    for c in cols:
        new_df = df.copy()
        new_df = new_df.sort_values(c, key=natsort_keygen())
        all_possible_varients[f'sort by {c}'] = new_df
    
    # select a column to sort 
    max_score = -float('inf')
    sort_by_col = cols[0]
    utterance_embedding = get_utterance_embedding(utterance.replace('?', ''), ft)
    for c in cols:
        col_embedding = get_utterance_embedding(c, ft)
        sim = get_embedding_cos_sim(utterance_embedding, col_embedding)
        if sim > max_score:
            max_score = sim
            sort_by_col = c
    new_df = df.copy()
    new_df = new_df.sort_values(sort_by_col, key=natsort_keygen())
    all_possible_varients[f'sort by max sim col {sort_by_col}({max_score})'] = new_df
    
    # select a column to sort 
    max_score = -float('inf')
    sort_by_col = cols[0]
    for c in cols:
        sim = pylcs.lcs_sequence_length(utterance, c) / len(c)
        if sim > max_score:
            max_score = sim
            sort_by_col = c
    new_df = df.copy()
    new_df = new_df.sort_values(sort_by_col, key=natsort_keygen())
    all_possible_varients[f'sort by max lcs col {sort_by_col}({max_score})'] = new_df
    
    # col pull to front
    for i in range(len(cols)):
        new_cols = [cols[i]] + cols[0:i] + cols[i+1:]
        new_df = df.copy()[new_cols]
        all_possible_varients[f'column pull front {cols[i]}'] = new_df
    
    # reorder the cols based on the col name overlap
    if utterance is not None:
        utterance = utterance.replace('?', '')
        occured_cols = []
        unoccured_cols = []
        for c in cols:
            if c.lower() in utterance:
                occured_cols.append(c)
            else:
                unoccured_cols.append(c)
        new_df = df.copy()[occured_cols + unoccured_cols]
        all_possible_varients[f'col reorder based on name overlaps'] = new_df
    
    # reorder the column based on the embedding distance to the utteracne
    if utterance is not None and ft is not None:
        utterance = utterance.replace('?', '')
        utterance_embedding = get_utterance_embedding(utterance, ft)
        
        overlap_counts = []
        for col in cols:
            col_embedding = get_utterance_embedding(col, ft)
            sim = get_embedding_cos_sim(utterance_embedding, col_embedding)
            overlap_counts.append(sim * (-1))
        
        new_cols = [cols[i] for i in np.argsort(overlap_counts).tolist()]
        new_df = df.copy()[new_cols]
        all_possible_varients[f'col reorder based on embedding distance'] = new_df

    # reorder the rows based on the value overlap
    if utterance is not None:
        utterance = utterance.replace('?', '')
        overlap_counts = []
        for i in range(df.shape[0]):
            count = 0
            values = df.iloc[i].tolist()
            for v in values:
                if str(v).lower() in utterance.lower():
                    count -= 1
            overlap_counts.append(count)
        new_df = df.copy()
        new_df = new_df.reindex(np.argsort(overlap_counts))
        all_possible_varients[f'row reorder based on value overlaps'] = new_df
        
    # reorder the rows based on the value embedding distance to the utteracne
    if utterance is not None and ft is not None:
        utterance = utterance.replace('?', '')
        utterance_embedding = get_utterance_embedding(utterance, ft)
        
        overlap_counts = []
        for i in range(df.shape[0]):
            values = ' '.join([str(v) for v in df.iloc[i].tolist()])
            value_embedding = get_utterance_embedding(values, ft)
            sim = get_embedding_cos_sim(utterance_embedding, value_embedding)
            overlap_counts.append(sim * (-1))
            
        new_df = df.copy()
        new_df = new_df.reindex(np.argsort(overlap_counts))
        all_possible_varients[f'row reorder based on embedding distance'] = new_df
    
    # reorder the rows based on the value overlap + reorder columns based on column name overlaps
    if utterance is not None:
        utterance = utterance.replace('?', '')
        overlap_counts = []
        for i in range(df.shape[0]):
            count = 0
            values = df.iloc[i].tolist()
            for v in values:
                if str(v).lower() in utterance.lower():
                    count -= 1
            overlap_counts.append(count)
        new_df = df.copy()
        new_df = new_df.reindex(np.argsort(overlap_counts))
        
        occured_cols = []
        unoccured_cols = []
        for c in cols:
            if c.lower() in utterance:
                occured_cols.append(c)
            else:
                unoccured_cols.append(c)
        new_df = new_df[occured_cols + unoccured_cols]
        all_possible_varients[f'row reorder based on value overlaps and col reorder based on name overlaps'] = new_df
        
    return all_possible_varients

def normalize_col_name(col_name, illegal_chars={'.': '', ' ':'_', 
                                                '\\':'_',  '(': '', 
                                                ')': '', '?': '', 
                                                '\n': '_', '&': '', 
                                                ':': '_', '/':'_', 
                                                ',': '_', '-': '_',
                                                'from': 'c_from',
                                                '\'': '',
                                                '%': 'percent',
                                                '#': 'num',
                                                '19': 'c_19', '20': 'c_20'}):
    for c in illegal_chars:
        col_name = col_name.replace(c, illegal_chars[c])
    col_name = re.sub('_+', '_', col_name)
    if re.search('\d', col_name[0]):
        col_name = 'c_' + col_name
    return col_name

def read_few_shot_demo(file_name, demo_num=None, at_index=None):
    with open(file_name, 'r') as f:
        all_demo = f.read()
    if demo_num is not None and at_index is None:
        return '\n\n'.join(all_demo.split('\n\n\n')[:demo_num])
    elif demo_num is None and at_index is not None:
        return all_demo.split('\n\n\n')[at_index]
    else:
        return all_demo.replace('\n\n\n', '\n\n')

class QuestionHandler:
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./'):
        self.qid = qid
        self.utterance = utterance
        self.source_csv = os.path.join(base_path, source_csv)
        self.target_value = target_value
        self.API_key = openai.api_key
        self._read_data()
        self.execution_acc = None
        self.execution_err = None
        self.predicted_sql = None
        self.reformat_sql = None
        self.gpt_error = None
        self.predicted_result = None
        self.original_output = None
        self.prompt = None
        
    def _read_data(self, ):
        self.source_table_df = pd.read_csv(self.source_csv, on_bad_lines='skip')
        # # print("Handler: ", self.source_table_df)
        self.source_schema = [normalize_col_name(c) for c in list(self.source_table_df.columns)]
        self.source_table_df.columns = self.source_schema
        self.data_examples = ''
        for i in range(min(100, self.source_table_df.shape[0])):
            self.data_examples += '\t'.join([str(i) for i in self.source_table_df.iloc[i].tolist()]) + '\n'
    
    def _log_dict(self):
        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'prompt': self.prompt if self.gpt_error is None else self.gpt_error,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            'df_columns': self.source_table_df.columns.tolist(),
            'gpt_original_output': self.original_output, 
        }

class CodexSQL(QuestionHandler):
    
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.demo_file = demo_file
        self.training_demo_ids = []
        self.frequency_penalty = 0.3
        self.model = "davinci-codex-002-msft"
    
    def _gen_codex_prompt(self, schema=True, demo_num=None, at_index=None):
        promp_template = """A database table "df" is shown as follows:
{}

Answer the following question with SQL based on the data above: "{}"

Therefore, the semantically and syntactically correct SQL query that answers the question is: ```"""
        
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df)
        
        self.prompt = promp_template.format(data_table, self.utterance)
        if self.demo_file:
            self.few_shot_demo = read_few_shot_demo(self.demo_file, demo_num=demo_num, at_index=at_index)
            self.prompt = self.few_shot_demo + '\n\n' + self.prompt
            
    
    def _get_gpt_prediction(self, ):
        openai.api_key = API_key
        try:
            # gpt = GPT(engine="davinci-codex-002-msft", # code-davinci-002  text-davinci-003
            #       temperature=0,
            #       max_tokens=1024,)
            self.original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=1024,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```')
            
            # output = openai.Completion.create(
            #   model="davinci-codex-002-msft",
            #   prompt=self.prompt,
            #   max_tokens=2000,
            #   temperature=0
            # )
            self.predicted_sql = self.original_output.choices[0].text.split('```')[0]
            
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_sql = None
            self.original_output = None
            
    def _evaluate_result(self, verbose=False):
        df = self.source_table_df
        # self.reformat_sql = self.predicted_sql.split('FROM')[0] + 'FROM df\nWHERE' + self.predicted_sql.split('WHERE')[-1]
        self.reformat_sql = self.predicted_sql
        try:
            output = sqldf(self.reformat_sql)
            if output.shape[0] > 0:
                result = str(output.iloc[0].tolist()[0])
                if result.strip().lower() in self.target_value.strip().lower():
                    self.execution_acc = True
                else:
                    self.execution_acc = False
                self.predicted_result = result
            else:
                self.predicted_result = []
            # if verbose:
                # print('===============================================')
                # print(f"Prompt: {self.prompt}")
                # print(f"Predicted SQL: {self.predicted_sql}")
                # print(f"Predicted results: {self.predicted_result}")
                # print(f"Expected results: {self.target_value}")
                # print(f"Execution math: {self.execution_acc}")
                # print('===============================================')
                
        except Exception as e:
            self.execution_acc = False
            self.execution_err = str(e)     
            # if verbose:
                # print(f"Exception encoutered: {e}")
            
    
class GptAnswer(QuestionHandler):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', \
                 demo_file=None, table_format=None, \
                 temperature=0, majority_vote=0):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.model = 'text-davinci-003'
        self.base_path = base_path
        self.target_value = target_value
        self.utterance = utterance
        self.source_csv = self.source_csv
        self.demo_file = demo_file
        self.demos = []
        self.prompt_template = \
"""A database table is shown as follows:
{}

Answer the following question based on the data above: "{}". The answer is: ```"""
        self.frequency_penalty = 0.3
        self.training_demo_ids = []
        self.temperature = temperature 
        self.table_format = table_format
        self.majority_vote = majority_vote
        assert self.majority_vote <= 0 or self.temperature > 0, 'If use majority_vote, please also specify a postive temperature.'
    
    def _gen_NN_demo(self, training_example_reasonings, training_embeddings, ft, demo_num=3):
        NNs_from_train = get_NN_demo(self.utterance, training_embeddings, ft, top_n=demo_num)
        for i in NNs_from_train:
            df = pd.read_csv(os.path.join(self.base_path, training_example_reasonings[i]['context']), on_bad_lines='skip')
            if self.table_format is not None:
                table_str = table_formater(df, 
                                           seperator=self.table_format['seperator'], 
                                           col_data_split=self.table_format['col_data_split'],
                                           col_prefix=self.table_format['col_prefix'],
                                           row_prefix=self.table_format['row_prefix'], 
                                           key_val_pairs=self.table_format['key_val_pairs']
                                          )
            else:
                table_str = table_formater(df)
            demo = self.prompt_template.format(table_str, training_example_reasonings[i]['utterance'])
            demo += training_example_reasonings[i]['targetValue'] + '```.'
            self.demos.append(demo)
        self.training_demo_ids = NNs_from_train
            
    def _gen_gpt_prompt(self, schema=True, demo_num=None, at_index=None):
        
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        if self.table_format is not None:
            data_table = table_formater(self.source_table_df, 
                                           seperator=self.table_format['seperator'], 
                                           col_data_split=self.table_format['col_data_split'],
                                           col_prefix=self.table_format['col_prefix'],
                                           row_prefix=self.table_format['row_prefix'],
                                           key_val_pairs=self.table_format['key_val_pairs']
                                          )
        else:
            data_table = table_formater(self.source_table_df)
            
        self.prompt = self.prompt_template.format(data_table, self.utterance)
        if len(self.demos) > 0:
            self.prompt = '\n\n'.join(self.demos) + '\n\n' + self.prompt
        elif self.demo_file:
            self.few_shot_demo = read_few_shot_demo(self.demo_file, demo_num=demo_num, at_index=at_index)
            self.prompt = self.few_shot_demo + '\n\n' + self.prompt
    
    def _get_gpt_prediction(self, ):
        openai.api_key = API_key
        
        try:
            self.original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```')
            
            self.predicted_result = self.original_output['choices'][0]['text'].replace('\n', '').strip(' ')
            
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_result = None
            # # print(self.qid, e)
    
    def _evaluate_result(self, verbose=False):
        if self.target_value.lower().strip(' ') in self.predicted_result.lower().strip(' '):
            self.execution_acc = True
        else:
            self.execution_acc = False
        return self.execution_acc 
    
    
    def _log_dict(self):
        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'prompt': self.prompt,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            # 'df_columns': self.source_table_df.columns.tolist(),
            'gpt_original_output': self.original_output, 
            'training_demo_ids': self.training_demo_ids
        }
            
class CodexAnswer(GptAnswer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.model = 'davinci-codex-002-msft'

class CodexAnswerNLTable(GptAnswer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)    
        self.model = 'davinci-codex-002-msft'
        
    def _set_table_nl_desc(self, table_nl_desc_train, table_nl_desc_test):
        self.table_nl_desc_train = table_nl_desc_train
        self.table_nl_desc_test = table_nl_desc_test
    
    def _gen_NN_demo(self, training_example_reasonings, training_embeddings, ft, demo_num=3):
        NNs_from_train = get_NN_demo(self.utterance, training_embeddings, ft, top_n=demo_num)
        for i in NNs_from_train:
            df = pd.read_csv(os.path.join(self.base_path, training_example_reasonings[i]['context']), on_bad_lines='skip')
            
            df, utterance = df, training_example_reasonings[i]['utterance']
            target_value = training_example_reasonings[i]['targetValue']
            
            # table_str = table_formater(df, utterance=utterance)
            table_str = self.table_nl_desc_train[i]['nl_text']
            if table_str is None:
                table_str = table_formater(df, utterance=utterance)
            
            demo = self.prompt_template.format(table_str, utterance)
            demo += target_value + '```.'
            self.demos.append(demo)
        self.training_demo_ids = NNs_from_train   
        
    def _gen_gpt_prompt(self, schema=True, demo_num=None, at_index=None):
        df, utterance, self.token_dict = tokenizeDFWithColNames(
            self.source_table_df, 
            self.utterance
        )
        # data_table = table_formater(df, utterance=utterance)
        data_table = self.table_nl_desc_test[int(self.qid.split('-')[-1])]['nl_text']
        if data_table is None:
            data_table = table_formater(df, utterance=utterance)
        self.prompt = self.prompt_template.format(data_table, utterance)
        if len(self.demos) > 0:
            self.prompt = '\n\n'.join(self.demos) + '\n\n' + self.prompt
    
    def _get_gpt_prediction(self, ):
        openai.api_key = API_key
        try:
            self.original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```')
            
            self.predicted_result = self.original_output['choices'][0]['text'].replace('\n', '').strip(' ')
            self.predicted_result = parseTokenizedStr(self.predicted_result, self.token_dict, False)
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_result = None

class CodexAnswerTokenizeDF(CodexAnswer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.model = 'davinci-codex-002-msft'
        
    def _gen_NN_demo(self, training_example_reasonings, training_embeddings, ft, demo_num=3):
        NNs_from_train = get_NN_demo(self.utterance, training_embeddings, ft, top_n=demo_num)
        for i in NNs_from_train:
            df = pd.read_csv(os.path.join(self.base_path, training_example_reasonings[i]['context']), on_bad_lines='skip')
            
            df, utterance, token_dict = tokenizeDFWithColNames(df, training_example_reasonings[i]['utterance'])
            target_value = parseTokenizedStr(str(training_example_reasonings[i]['targetValue']), token_dict, True)
            
            # df, utterance = df, training_example_reasonings[i]['utterance']
            # target_value = training_example_reasonings[i]['targetValue']
            
            table_str = table_formater(df, utterance=utterance)
            demo = self.prompt_template.format(table_str, utterance)
            demo += target_value + '```.'
            self.demos.append(demo)
        self.training_demo_ids = NNs_from_train   
        
    def _gen_gpt_prompt(self, schema=True, demo_num=None, at_index=None):
        df, utterance, self.token_dict = tokenizeDFWithColNames(
            self.source_table_df, 
            self.utterance
        )
        data_table = table_formater(df, utterance=utterance)
        self.prompt = self.prompt_template.format(data_table, utterance)
        if len(self.demos) > 0:
            self.prompt = '\n\n'.join(self.demos) + '\n\n' + self.prompt
    
    def _get_gpt_prediction(self, ):
        openai.api_key = API_key
        try:
            self.original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```')
            
            self.predicted_result = self.original_output['choices'][0]['text'].replace('\n', '').strip(' ')
            self.predicted_result = parseTokenizedStr(self.predicted_result, self.token_dict, False)
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_result = None
    
        
# use the frequence penalty
class GptAnswerReason(GptAnswer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.demo_file = demo_file
        self.model = 'text-davinci-002'
        self.frequency_penalty = 0.7
        
    def _read_data(self, ):
        self.source_table_df = pd.read_csv(self.source_csv, on_bad_lines='skip')
        self.source_schema = [normalize_col_name(c) for c in list(self.source_table_df.columns)]
        self.source_table_df.columns = self.source_schema
        self.data_examples = ''
        for i in range(min(100, self.source_table_df.shape[0])):
            self.data_examples += '\t'.join([str(i) for i in self.source_table_df.iloc[i].tolist()]) + '\n'
    
    def _gen_gpt_prompt(self, schema=True, demo_num=3):
        promp_template = """{}


A database table is shown as follows:
{}

Answer the following question based on the data above: "{}". Generate step-by-step reasoning.

Reasoning steps:
"""
        self.few_shot_demo = read_few_shot_demo(self.demo_file, demo_num=demo_num)
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df)
        self.prompt = promp_template.format(self.few_shot_demo, data_table, self.utterance)
    
    def _get_gpt_prediction(self, ):
        openai.api_key = API_key
        try:
            # gpt = GPT(engine= self.model, # code-davinci-002  text-davinci-003 "davinci-code-002-msft"
            #       temperature=0,
            #        max_tokens=1024)
            # self.original_output = gpt.submit_request(self.prompt, frequency_penalty=self.frequency_penalty)
            self.original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```.')
            
            self.original_result = self.original_output['choices'][0]['text']
            if "```" in self.original_result:
                self.predicted_result = self.original_result.split('```')[1].split('```')[0]
            else:
                self.predicted_result = self.original_result.strip('\n').strip(' ').strip('.').split('.')[-1]
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_result = None
            # print(self.qid, e)

class CodexAnswerReason(GptAnswerReason):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path, demo_file)
        self.model = 'davinci-codex-002-msft'
            

class CodexSQLReason(CodexSQL):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.demo_file = demo_file
        self.model = 'davinci-codex-002-msft'
        self.frequency_penalty = 0.3
        
    # def _read_data(self, ):
    #     self.source_table_df = pd.read_csv(self.source_csv, on_bad_lines='skip')
    #     self.source_schema = [normalize_col_name(c) for c in list(self.source_table_df.columns)]
    #     self.source_table_df.columns = self.source_schema
    #     self.data_examples = ''
    #     for i in range(min(100, self.source_table_df.shape[0])):
    #         self.data_examples += '\t'.join([str(i) for i in self.source_table_df.iloc[i].tolist()]) + '\n'
    
    def _gen_gpt_prompt(self, schema=True, demo_num=3):
        promp_template = """{}
 
A database table is shown as follows:
{}

Answer the following question with SQL based on the data above: "{}". Generate step-by-step reasoning.

Therefore, the semantically and syntactically correct SQL query that answers the question is:```"""
        self.few_shot_demo = read_few_shot_demo(self.demo_file, demo_num=demo_num)
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df)
        self.prompt = promp_template.format(self.few_shot_demo, data_table, self.utterance)
    
    def _get_gpt_prediction(self, ):
        openai.api_key = API_key
        try:
            # gpt = GPT(engine= self.model,
            #       temperature=0,
            #        max_tokens=1024)
            # self.original_output = gpt.submit_request(self.prompt, stop='```', frequency_penalty=self.frequency_penalty)
            
            self.original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=1024,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```')
            
            self.original_result = self.original_output['choices'][0]['text']
            if "SELECT" in self.original_result:
                self.predicted_sql = self.original_result.replace('\n', ' ')
            else:
                self.predicted_result = self.original_result.strip('\n').strip(' ').strip('.').split('.')[-1]
                self.predicted_sql = None
            # # print(self.predicted_result)
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_sql = None
            self.predicted_result = None
            # print(self.qid, e)
            
class CodexAnswerOrderExplorer(CodexAnswer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.all_gpt_predicted_results = {}
        self.execution_acc = False
        self.predicted_result = None
        self.prompts = {}
    
    def _gen_all_table_permutations(self, ft=None):
        self.dataframe_permutations = permuteDataFrame(self.source_table_df, self.utterance, ft)
    
    def _gen_NN_demo(self, training_example_reasonings, training_embeddings, ft, demo_num=3):
        NNs_from_train = get_NN_demo(self.utterance, training_embeddings, ft, top_n=demo_num)
        for i in NNs_from_train:
            df = pd.read_csv(os.path.join(self.base_path, training_example_reasonings[i]['context']), on_bad_lines='skip')
            if self.table_format is not None:
                table_str = table_formater(df, 
                                           seperator=self.table_format['seperator'], 
                                           col_data_split=self.table_format['col_data_split'],
                                           col_prefix=self.table_format['col_prefix'],
                                           row_prefix=self.table_format['row_prefix'], 
                                           key_val_pairs=self.table_format['key_val_pairs']
                                          )
            else:
                table_str = table_formater(df)
            demo = self.prompt_template.format(table_str, training_example_reasonings[i]['utterance'])
            demo += training_example_reasonings[i]['targetValue'] + '```.'
            self.demos.append(demo)
        self.training_demo_ids = NNs_from_train
    
    def _gen_gpt_prompt(self, schema=True, demo_num=None, at_index=None):
        
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df, permute_df=False)
        self.prompt = self.prompt_template.format(data_table, self.utterance)
        if len(self.demos) > 0:
            self.prompt = '\n\n'.join(self.demos) + '\n\n' + self.prompt
    
    def _get_gpt_prediction(self, permutation_method):
        openai.api_key = API_key
        try:
            self.prompts[permutation_method] = self.prompt
            original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```')
            prediction = original_output['choices'][0]['text'].replace('\n', '').strip(' ')
            if '=' in prediction:
                prediction = prediction.split('=')[-1]
            if ':' in prediction:
                prediction = prediction.split(':')[-1]
                
            self.all_gpt_predicted_results[permutation_method] = prediction.strip(' ')
            self.predicted_result = self.all_gpt_predicted_results[permutation_method]
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_result = None
            
    def _explore_all_dataframe_permutations(self,):
        for permutation_method in self.dataframe_permutations:
            self.source_table_df = self.dataframe_permutations[permutation_method]
            self._gen_gpt_prompt()
            self._get_gpt_prediction(permutation_method)
            if self.gpt_error is not None:
                break
    
    
    def _log_dict(self):
        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'prompt': self.prompts if self.gpt_error is None else self.gpt_error,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            'df_columns': self.source_table_df.columns.tolist(),
            'gpt_original_output': self.original_output, 
            'all_df_permutation_results': self.all_gpt_predicted_results
        }

    
class CodexAnswerRandShuffle(CodexAnswerOrderExplorer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None):
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.all_gpt_predicted_results = {}
        self.execution_acc = False
        self.predicted_result = None
        self.prompts = {}
        
    def _gen_all_table_permutations(self, ft=None):
        self.dataframe_permutations = shuffleDataFrame(self.source_table_df, n=10)
        
        
