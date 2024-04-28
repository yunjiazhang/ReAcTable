import pandas as pd
import openai
import os
import json
from tqdm import tqdm
import dotenv

from dateutil import parser
from tabqa.GptPrompter import *
from tabqa.GptCOTPrompter import *
from collections import Counter


def get_token_num(s):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(s, add_special_tokens=False)
    num_tokens = len(tokens)
    return num_tokens

class CodexAnswerCOTExecutor_HighTemperaturMajorityVote(CodexAnswerCOTExecutor_template):
    def __init__(self, prompt_template_json, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        super().__init__(prompt_template_json, qid, utterance, source_csv, target_value, base_path=base_path, demo_file=demo_file, sep=sep)
        self.temperature = 0.6
    
    def _get_gpt_prediction_majority_vote(self, NNDemo=False, ft=None, repeat_times=5, maintain_df_ids=False):
        all_predictions = []
        for _ in range(repeat_times):
            self._read_data()
            self._gen_gpt_prompt(NNDemo, ft, maintain_df_ids=maintain_df_ids)
            self._get_gpt_prediction(maintain_df_ids=maintain_df_ids)
            all_predictions.append(self.predicted_result)
        self.all_predictions = all_predictions
        from collections import Counter
        counter = Counter(all_predictions)
        majority = counter.most_common(1)[0][0]
        self.predicted_result = majority
    
    def _log_dict(self):
        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'prompt': self.prompt,
            # 'action_prompts': self.action_prompts,
            # 'decision_prompts': self.decision_prompts,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            'gpt_original_output': self.original_output, 
            'all_predictions': self.all_predictions,
            'training_demo_ids': self.training_demo_ids
        }
       
    
class CodexAnswerCOTExecutor_BeamSeach(CodexAnswerCOTExecutor_template):
    def __init__(self, prompt_template_json, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        super().__init__(prompt_template_json, qid, utterance, source_csv, target_value, base_path=base_path, demo_file=demo_file, sep=sep)
        self.temperature = 0.6
        # tune hyperparameters
    
    def _top_k_keys(self, d, k):
        """
        Return the keys with the top-k values in a dictionary d.
        """
        # Sort the items in the dictionary by their values in descending order.
        items = sorted(d.items(), key=lambda x: x[1][0], reverse=True)

        # Get the first k items and extract the keys.
        top_k_items = items[:k]
        top_k_keys = [item[0] for item in top_k_items]

        return top_k_keys
    
    def _get_gpt_prediction(self, beam_size=2):
        self.original_output = []
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()] 
        
        current_brunches = [[(self.prompt, self.source_table_df, 0),]]
        self.all_predictions = []
        
        iteration_cnt = 0
        while True:
            if len(current_brunches[-1]) == 0:
                break
            
            iteration_cnt += 1
            # answers_df = {}
            answers_df = []
            
            for current_prompt, df, score in current_brunches[-1]:
                original_output = GptCompletion(engine=self.model,
                                                prompt=current_prompt,
                                                max_tokens=128,
                                                temperature=self.temperature,
                                                top_p=1,
                                                frequency_penalty=self.frequency_penalty,
                                                n=beam_size,
                                                logprobs=1,
                                                stream=False,
                                                stop='```.',
                                                best_of=beam_size)
                # print(original_output)
                for i in range(len(original_output['choices'])):
                    original_result = original_output['choices'][i]['text'].strip('\n')
                    logprobs = sum(original_output['choices'][i]['logprobs']['token_logprobs'])
                    answers_df.append((original_result, 0, df, current_prompt))
                    # if original_result in answers_df:
                    #     answers_df[original_result] = (score+max(logprobs, answers_df[original_result][0]), df, current_prompt)
                    # else:
                    #     answers_df[original_result] = (score+logprobs, df, current_prompt)
            
            # topk_answers = self._top_k_keys(answers_df, beam_size)
            # topk_answers = list(answers_df.keys())
            
            # print(topk_answers)
            
            current_branches.append([])
            
            for sample in answers_df:
                # answer, current_score, current_df, current_prompt = answers_df[answer]
                answer, current_score, current_df, current_prompt = sample
                answer_type = answer.split(':')[0].replace('\n', '').replace(' ', '')
                answer = answer.split('```')[1]  
                # print("Processing: ", answer)
                
                if answer_type == 'Answer':
                    # self.all_predictions.append(answer.split('```')[-1])
                    self.all_predictions.append(answer.split('```')[-1])
                    # if answer.split('```')[-1] in self.all_predictions:
                    #     self.all_predictions[answer.split('```')[-1]] = max(current_score, self.all_predictions[answer.split('```')[-1]])
                    # else:
                    #     self.all_predictions[answer.split('```')[-1]] = current_score
                    continue
                elif answer_type in self.supported_code_types:
                    renewed_df = self._executor(current_df, answer, answer_type)
                    
                    if renewed_df is None:
                        renewed_df = self._executor(self.source_table_df, answer, answer_type)
                        
                    if renewed_df is None:
                        # self._gen_gpt_prompt()
                        if len(self.all_predictions) == 0:
                            tmp_prompt = current_prompt.strip('\n') + '\n\nAnswer: ```'
                            original_output = GptCompletion(engine=self.model,
                                                    prompt=tmp_prompt,
                                                    max_tokens=128,
                                                    temperature=self.temperature,
                                                    top_p=1,
                                                    frequency_penalty=self.frequency_penalty,
                                                    n=1,
                                                    stream=False,
                                                    stop='```.')
                            original_result = original_output['choices'][0]['text'].replace('\n', '')
                            # self.all_predictions.append(original_result)
                            self.all_predictions.append(original_result)
                            # if original_result in self.all_predictions:
                            #     self.all_predictions[original_result] = max(current_score, self.all_predictions[original_result])
                            # else:
                            #     self.all_predictions[original_result] = current_score
                        continue   

                    data_table = table_formater(renewed_df, permute_df=False, line_limit=self.line_limit)
                    intermediate_prompt_template = self.prompt_template_dict['intermediate_prompt_template'][answer_type]
                    tmp_prompt = current_prompt.strip('\n') + '\n\n' + original_result + \
                        '```.\n\n' + intermediate_prompt_template.format(data_table, self.utterance)
                    # if answer_type == 'Python':
                    #     print(tmp_prompt)
                    current_brunches[-1].append((tmp_prompt, renewed_df, current_score))

                else:
                    self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
                    current_prompt = current_prompt.strip('\n') + '\n\nAnswer: ```'
                    original_output = GptCompletion(engine=self.model,
                                                prompt=current_prompt,
                                                max_tokens=128,
                                                temperature=self.temperature,
                                                top_p=1,
                                                frequency_penalty=self.frequency_penalty,
                                                n=1,
                                                stream=False,
                                                stop='```.')
                    original_result = original_output['choices'][0]['text'].replace('\n', '')
                    # self.all_predictions.append(original_result)
                    self.all_predictions.append(original_result)
                    # if original_result in self.all_predictions:
                    #     self.all_predictions[original_result] = max(current_score, self.all_predictions[original_result])
                    # else:
                    #     self.all_predictions[original_result] = current_score
        
        # print(self.all_predictions)
        
        counts = Counter(self.all_predictions)
        most_common = counts.most_common(1)[0][0]
        self.predicted_result = most_common
        
        
    
    def _log_dict(self):
        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'prompt': self.prompt,
            # 'action_prompts': self.action_prompts,
            # 'decision_prompts': self.decision_prompts,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            'gpt_original_output': self.original_output, 
            'all_predictions': self.all_predictions,
            'training_demo_ids': self.training_demo_ids
        }
    

class CodexAnswerCOTExecutor_LeverVote(CodexAnswerCOTExecutor_template):
    def __init__(self, prompt_template_json, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        super().__init__(prompt_template_json, qid, utterance, source_csv, target_value, base_path=base_path, demo_file=demo_file, sep=sep)
        self.temperature = 0.6
        # self.line_limit = 20 if 'code' in self.model else 10
        self.line_limit = 10
        # tune hyperparameters
        
    def dataframe_is_subset(self, df1, df2):
        
        if len(df1.columns) > len(df2.columns):
            df1, df2 = df2, df1
        
        column_set1 = set(df1.columns)
        column_set2 = set(df2.columns)
        if df1.shape[0] != df2.shape[0] or df1.shape[0]==0 or df2.shape[0]==0:
            return False
        
        if column_set1 == column_set2:
            trans_column_set2 = df2[list(df1.columns)]
        else:
            trans_column_set2 = df2
        
        sentence_set1 = set()
        sentence_set2 = set()
        
        for index, row in df1.iterrows():
            sentence = ''
            for c in df1.columns:
                sentence += f'{row[c]}'
            sentence_set1.add(sentence)
        
        for index, row in trans_column_set2.iterrows():
            sentence = ''
            for c in trans_column_set2.columns:
                sentence += f'{row[c]}'
            sentence_set2.add(sentence)
            
        return sentence_set1 == sentence_set2
    
    def add_to_outcomes(self, outcomes, answer_type, answer, result, logprob):
        for idx, (out_ans_type, out_ans, out_df, out_logprob) in enumerate(outcomes):
            if type(result) == str and type(out_df) == str and result == out_df:
                # outcomes[idx] = (answer_type, answer, result, np.log(np.exp(logprob)+np.exp(outcomes[idx][3])))
                outcomes[idx] = (answer_type, answer, result, max(logprob, outcomes[idx][3]))
                return
            elif type(result) != str and type(out_df) != str and self.dataframe_is_subset(out_df, result):
                # outcomes[idx] = (answer_type, answer, result, np.log(np.exp(logprob)+np.exp(outcomes[idx][3])))
                outcomes[idx] = (answer_type, answer, result, max(logprob, outcomes[idx][3]))
                return
        
        if 'Python' in answer_type:
            logprob += 100
        
        if type(result) != str and result.shape[0] == 0:
            logprob = float('-inf')
            
        outcomes.append((answer_type, answer, result, logprob))
        return
    
    def select_next_step(self, outcomes):
        return max(outcomes, key=lambda x: x[3])
    
    def _get_gpt_prediction(self, trial_cnt=5):
        self.original_output = []
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()] 
        
        current_brunches = [[(self.prompt, self.source_table_df, 0),]]
        self.all_predictions = []
        
        iteration_cnt = 0
        while True:
            # print("Start step: ===============")
            iteration_cnt += 1
            if iteration_cnt > self.iteration_max_limit:
                self.prompt += '\nAnswer: ```'
                original_output = GptCompletion(engine=self.model,
                                        prompt=self.prompt.strip('\n'),
                                        max_tokens=128,
                                        temperature=self.temperature,
                                        top_p=1,
                                        frequency_penalty=self.frequency_penalty,
                                        n=1,
                                        stream=False,
                                        # stop='```.'
                                        prompt_end=None
                                        )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
                break
            
            self.prompts.append(self.prompt)
            generated_results = []
            self.original_output.append([])
            # print('========================================')
            # print(self.prompt)
            # print('========================================')
            
            original_output = GptCompletion(engine=self.model,
                                                prompt=self.prompt,
                                                max_tokens=128,
                                                temperature=self.temperature,
                                                top_p=1,
                                                frequency_penalty=self.frequency_penalty,
                                                n=trial_cnt,
                                                logprobs=1,
                                                stream=False,
                                                best_of=trial_cnt)
            for idx in range(len(original_output['choices'])):
                original_result = original_output['choices'][idx]['text'].strip('\n')
                answer_type = original_result.split(":")[0].replace('\n', '').replace(' ', '')
                answer = original_result.split('```')[-1]
                # logprobs = sum(original_output['choices'][idx]['logprobs']['token_logprobs'])
                logprobs = sum(original_output['choices'][idx]['logprobs']['token_logprobs']) \
                         / len(original_output['choices'][idx]['logprobs'])
                
                if answer_type == 'Answer':
                    predicted_result = answer.split('```')[-1]
                    self.add_to_outcomes(generated_results, answer_type, answer, predicted_result, logprobs)
                elif answer_type in self.supported_code_types:
                    renewed_df = self._executor(self.source_table_df.copy(), answer, answer_type)

                    i = len(self.series_dfs) - 1
                    while i >= 0 and (renewed_df is None): # or renewed_df.shape[0] == 0):
                        self.source_table_df = self.series_dfs[i]
                        renewed_df = self._executor(self.source_table_df, answer, answer_type)
                        if renewed_df is not None:
                            self.gpt_error = None
                        i -= 1
                    
                    if renewed_df is None:
                        # self._gen_gpt_prompt()
                        tmp_prompt = self.prompt + '\nAnswer: ```'
                        tmp_original_output = GptCompletion(engine=self.model,
                                                prompt=tmp_prompt,
                                                max_tokens=128,
                                                temperature=self.temperature,
                                                top_p=1,
                                                frequency_penalty=self.frequency_penalty,
                                                n=1,
                                                stream=False,
                                                # stop='```.'
                                                prompt_end=None
                                                )
                        tmp_original_result = tmp_original_output['choices'][0]['text'].replace('\n', '')
                        tmp_predicted_result = tmp_original_result
                        self.add_to_outcomes(generated_results, 
                                             'Answer', 
                                             tmp_predicted_result, 
                                             tmp_predicted_result, 
                                             logprobs-100) # penalize the score with broken type
                    else:
                        # print("New df: ", renewed_df)
                        self.add_to_outcomes(generated_results, answer_type, answer, renewed_df, logprobs)  
                        # self.original_output[-1].append(answer)

                else:
                    # if len(generated_results) == 0:
                    self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
                    tmp_prompt = self.prompt + '\nAnswer: ```'
                    tmp_original_output = GptCompletion(engine=self.model,
                                            prompt=tmp_prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            prompt_end=None
                                            )
                    tmp_original_result = tmp_original_output['choices'][0]['text'].replace('\n', '')
                    tmp_predicted_result = tmp_original_result
                    self.add_to_outcomes(generated_results, 'Answer', tmp_predicted_result, tmp_predicted_result, logprobs-100) # penalize the score with broken type
            
            code_type, code, result, _ = self.select_next_step(generated_results)
            self.original_output.append([(l[-1], l[1]) for l in generated_results])
            self.original_output[-1].append((1, code))
            
            # =================================
            # print(generated_results)
            # print(f"SELECTED: {code_type}: {code} ")
            # =================================
            
            if type(result) == str:
                self.predicted_result = result
                self.prompt = self.prompt.strip('\n') + '\n\n' + code_type + ': ```' + code + '```.'
                break
            else:
                self.source_table_df = result
                data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
                intermediate_prompt_template = self.prompt_template_dict['intermediate_prompt_template'][answer_type]
                self.prompt = self.prompt.strip('\n') + '\n\n' + code_type + ': ```' + code + '```.\n\n' + intermediate_prompt_template.format(data_table, self.utterance)
                self.series_dfs.append(result)
    
    def _log_dict(self):
        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'prompt': self.prompt,
            # 'action_prompts': self.action_prompts,
            # 'decision_prompts': self.decision_prompts,
            'execution_match': self.execution_acc,
            'gpt_error': self.gpt_error,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            'gpt_original_output': self.original_output, 
            'all_predictions': self.all_predictions,
            'training_demo_ids': self.training_demo_ids
        }