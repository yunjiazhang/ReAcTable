import re
import pandas as pd
from openai import OpenAI
import openai
client = OpenAI(api_key=openai.api_key)
import os
import json
# from gpt3_sandbox.api.gpt import GPT
# from gpt3_sandbox.api.gpt import Example
from pandasql import sqldf
from tqdm import tqdm
import numpy as np
from tabqa.GptPrompter import *
import fasttext.util

class GptReasoner():

    def __init__(self, qid, utterance, source_csv_file, target_value, base_path='./dataset/WikiTableQuestions/'):
        self.prompt_template = \
"""
A database table is shown as follows:
{}

Answer the following question based on the data above: "{}". The answer is:"{}".

Here is the step-by-step reasoning:```  
"""
        self.model = 'text-davinci-002'
        self.max_tokens = 1024
        self.temperature = 0
        self.n = 1
        self.top_p = 1
        self.stop = '```'
        self.base_path = base_path
        self.qid = qid
        self.source_csv_file = os.path.join(base_path, source_csv_file)
        self.utterance = utterance
        self.target_value = target_value
        self.gpt_error = None
        self.gpt_reasoning = None

    def _generate_reasoning_prompt(self, ):
        self.source_df = pd.read_csv(self.source_csv_file, on_bad_lines='warn')
        table_str = table_formater(self.source_df, seperator='|', col_data_split='-')
        self.prompt = self.prompt_template.format(table_str, self.utterance, self.target_value)

    def generate_reasonings(self, ):
        try:
            self._generate_reasoning_prompt()
            self.gpt_reasoning = client.completions.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=self.max_tokens,
                                            temperature=self.temperature,
                                            top_p=self.top_p,
                                            n=self.n,
                                            stream=False,
                                            stop=self.stop).choices[0].text.strip('\n').strip(' ')
        except Exception as e:
            self.gpt_error = str(e)


class GptAutoReasoner(QuestionHandler):

    def __init__(self, qid, utterance, source_csv_file, target_value, base_path='./dataset/WikiTableQuestions/'):
        super().__init__(qid, utterance, source_csv_file, target_value, base_path)
        self.prompt_template = \
"""
A database table is shown as follows:
{}

Answer the following question based on the data above: "{}". Generate step-by-step reasoning.

Reasoning steps: """
        self.demos = []
        # self.model = 'text-davinci-002'
        self.model = 'davinci-codex-002-msft'
        self.max_tokens = 1024
        self.temperature = 0
        self.n = 1
        self.top_p = 1
        self.stop = '```.'
        self.base_path = base_path
        self.source_csv_file = source_csv_file
        self.frequency_penalty = 0.3
        # self.frequency_penalty = 0

    def _gen_NN_demo(self, training_example_reasonings, training_embeddings, ft, demo_num=3):
        NNs_from_train = get_NN_demo(self.utterance, training_embeddings, ft, top_n=demo_num)
        for i in NNs_from_train:
            df = pd.read_csv(os.path.join(self.base_path, training_example_reasonings[i]['context']), on_bad_lines='skip')
            table_str = table_formater(df)
            demo = self.prompt_template.format(table_str, training_example_reasonings[i]['utterance'])
            demo += str(training_example_reasonings[i]['gptReasoning']).replace('\n', ' ')
            demo += ' Therefore, the answer is: ```' + training_example_reasonings[i]['targetValue'] + '```.'
            self.demos.append(demo)
        self.training_demo_ids = NNs_from_train


    def _gen_gpt_prompt(self, ):
        source_csv = pd.read_csv(os.path.join(self.base_path, self.source_csv_file), on_bad_lines='skip')
        table_str = table_formater(source_csv)
        self.prompt = self.prompt_template.format(table_str, self.utterance)
        if len(self.demos) > 0:
            self.prompt = '\n\n'.join(self.demos) + self.prompt

    def _get_gpt_prediction(self, ):
        try:
            self.gpt_original_output = client.completions.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=self.max_tokens,
                                            temperature=self.temperature,
                                            top_p=self.top_p,
                                            frequency_penalty=self.frequency_penalty,
                                            n=self.n,
                                            stream=False,
                                            stop=self.stop).choices[0].text.strip('\n').strip(' ') + self.stop
            if "```" in self.gpt_original_output:
                self.predicted_result = self.gpt_original_output.split('```')[-2].replace('\n', ' ').strip(' ')
            else:
                self.predicted_result = self.gpt_original_output.split('.')[-2].replace('\n', ' ').strip(' ')
        except Exception as e:
            self.gpt_error = str(e)
            self.predicted_result = ''
            self.gpt_original_output = None
            # print(self.qid, e)

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
            'gpt_original_output': self.gpt_original_output,
            'prompt': self.prompt,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            'df_columns': self.source_table_df.columns.tolist(),
            'training_demo_ids': self.training_demo_ids
        }




