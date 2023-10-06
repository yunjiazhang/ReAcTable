from dateutil import parser
from tabqa.GptCOTPrompter import *
from collections import Counter
# from openai import tiktoken

class CodexAnswerCOTExecutor_SplitFact(CodexAnswerCOTExecutor_template):
    def __init__(self, 
                 prompt_template_json, 
                 qid, 
                 utterance, 
                 source_csv, 
                 target_value, 
                 base_path='./', 
                 demo_file=None, 
                 sep=',', 
                 splitfact_template_json=None, 
                 splitfact_demo_file=None,
                 splitfact_model=None,
                 table_title=None,
                 verbose=False,
                 method_name=None
                ):
        super().__init__(
            prompt_template_json, 
            qid, 
            utterance, 
            source_csv, 
            target_value, 
            base_path=base_path, 
            demo_file=demo_file, 
            sep=sep)
        self.line_limit = 20 if 'code' in self.model else 10
        self.splitfact_template_json = splitfact_template_json
        self.splitfact_demo_file = splitfact_demo_file
        self.splitfact_model = splitfact_model
        self.prompts = []
        self.temperature = 0
        self.table_title = table_title
        self.iteration_max_limit = float('inf')
        self.method_name = method_name
        self.verbose = verbose
        self.max_token_limit = 8000

    def _split_facts(self, fact_string):
        """
        The fact string looks like: 
        SplitFact: ```(Fact statement 1) AND (Fact statement 2)```.
        """
        pattern = r'\((.*?)\)'
        matches = re.findall(pattern, fact_string)
        return matches
    
    def _gen_gpt_prompt(self, maintain_df_ids=False):
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
        
        if self.table_title is None:
            self.prompt = self.prompt_template.format(data_table, self.utterance)
        else:
            self.prompt = self.prompt_template.format(self.table_title, data_table, self.utterance)
        
        if maintain_df_ids:
            self.prompt = self.prompt.replace("DF", "DF0")
        
        # format demo
        assert '.json' in self.demo_file, "Use json file as the demo file format"
        self.demo_prompt = ""
        demos = json.load(open(os.path.join(self.base_path, self.demo_file)))
        
        if self.demo_ids is not None:
            demos = [demos[i] for i in self.demo_ids]

        if not maintain_df_ids:
            for demo in demos[0:self.max_demo]:
                for i in range(len(demo['tables'])):
                    if i == 0:
                        if self.table_title is None:
                            self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']) + '\n\n'
                        else:
                            self.demo_prompt += self.prompt_template.format(demo['tableTitle'], demo['tables'][i], demo['utterance']) + '\n\n'
                            
                    else:
                        if 'SQL:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['SQL'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                        elif 'Python:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['Python'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                    if self.method_name is not None:
                        self.demo_prompt += self.method_name + ': ' + demo['responses'][i] + '\n\n'
                    else:
                        self.demo_prompt += demo['responses'][i] + '\n\n'
        else:
            for demo in demos[0:self.max_demo]:
                for i in range(len(demo['tables'])):
                    if i == 0:
                        if self.table_title is None:
                            self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']).replace(' DF ', ' DF0 ') + '\n\n'
                        else:
                            self.demo_prompt += self.prompt_template.format(demo['tableTitle'], demo['tables'][i], demo['utterance']).replace(' DF ', ' DF0 ') + '\n\n'
                    else:
                        if 'SQL:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['SQL'].replace(":\n", f" (DF{i}):\n").format(demo['tables'][i], demo['utterance']) + '\n\n'
                        elif 'Python:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['Python'].replace(":\n", f" (DF{i}):\n").format(demo['tables'][i], demo['utterance']) + '\n\n'
                    if self.method_name is not None:
                        self.demo_prompt += self.method_name + ': ' + demo['responses'][i] + '\n\n'
                    else:
                        self.demo_prompt += demo['responses'][i] + '\n\n'
        
        self.prompt = self.demo_prompt + self.prompt + '\n'
        
        # token_count = tiktoken.token_count(prompt)
        # if token_count > self.max_token_limit:
        #     self.max_demo = max(0, self.max_demo-2)
        #     if self.max_demo <= 0:
        #         return 
        #     else:
        #         self._gen_gpt_prompt(maintain_df_ids)
            
        
        
    def _get_gpt_prediction_splitfact(self, 
                                      maintain_df_ids=False,
                                      split_fact=True,
                                      early_stop_when_table_has_answer=True
                                     ):
        # =========================
        # First split the facts if necessary
        # =========================
        if split_fact:
            self.original_output = []
            self._get_split_facts()
            if self.verbose:
                print("original statement: ", self.utterance)
                print("converted into: ", self.unit_fact_statements)
            self.original_output.append("SplitFact: ```" + self.original_fact_eval_string + '```.')
        else:
            self.original_fact_eval_string = f'({self.utterance})'
            self.unit_fact_statements = [self.utterance]
        
        # =========================
        # Second, verify each fact statement
        # =========================
        original_utterance = self.utterance
        self.fact_statement_checking_results = []
        fact_eval_string = self.original_fact_eval_string
        for unit_f in self.unit_fact_statements:
            self._read_data() # re-read the data from csv to make sure we start eval from scratch
            if self.verbose:
                print(f"checking: {unit_f}")
            self.original_output.append(f"...checking: {unit_f}.")
            self.utterance = unit_f + '. yes or no?'
            self._gen_gpt_prompt(maintain_df_ids=maintain_df_ids)
            self._get_gpt_prediction(maintain_df_ids=maintain_df_ids, 
                                     early_stop_when_table_has_answer=early_stop_when_table_has_answer)
            fact_eval_string = fact_eval_string.replace(
                f'({unit_f})',
                ' True ' if 'yes' in self.predicted_result \
                         else ' False ')

        self.utterance = original_utterance
        # =========================
        # finally merge the facts 
        # =========================
        if eval(fact_eval_string):
            self.predicted_result = 'yes'
        else:
            self.predicted_result = 'no'
        self.original_output.append(
            f"MergeAnswer: {fact_eval_string} = {self.predicted_result}.")

    def _get_split_facts(self, ):
        
        # =========================
        # Engage GPT to decide the split
        # =========================
        with open(self.base_path + '/' + self.splitfact_template_json, 'r') as f:
            prompt_template = json.load(f)['prompt_template']
        
        with open(self.base_path + '/' + self.splitfact_demo_file, 'r') as f:
            splitfact_demos = json.load(f)
        
        # Demos 
        self.splitfact_prompt = []
        for demo in splitfact_demos:
            self.splitfact_prompt.append(
                prompt_template.format(
                    demo['utterance'].replace(' yes or no?', '').strip('.')) \
                      + f'\n\nConverted: ```{demo["splitted"]}```.')
        
        data_table = table_formater(
            self.source_table_df, 
            permute_df=False, 
            line_limit=self.line_limit)
        self.splitfact_prompt.append(
            prompt_template.format(self.utterance.replace(' yes or no?', '').strip('.')))
        self.splitfact_prompt = '\n\n'.join(self.splitfact_prompt) + '\n\nConverted: ```'
        fact_string = GptCompletion(
                engine=self.splitfact_model\
                        if self.splitfact_model is not None\
                        else self.model,
                prompt=self.splitfact_prompt,
                max_tokens=128,
                temperature=self.temperature,
                top_p=1,
                frequency_penalty=self.frequency_penalty,
                n=1,
                stream=False,
                debug=self.verbose
            )['choices'][0]['text']

        self.original_fact_eval_string = fact_string
        self.unit_fact_statements = self._split_facts(fact_string)
        
    def _get_gpt_prediction(self, maintain_df_ids=False, early_stop_when_table_has_answer=False):
        
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()]        
        
        self.code_history = []
        
        iteration_cnt = 0
        while True:
            iteration_cnt += 1
            
            if self.method_name is not None:
                self.prompt = self.prompt.strip('\n') + f'\n\n{self.method_name}:'
            else:
                self.prompt = self.prompt.strip('\n') + '\n\n'
                
            original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.',
                                            debug=self.verbose,
                                           )
            
            original_result = original_output['choices'][0]['text'].strip('\n')
            answer_type = original_result.split(":")[0].replace('\n', '').replace(' ', '')
            answer = original_result.split('```')[-1]
            self.original_output.append(original_result)
            self.prompts.append(self.prompt + original_result)
            
            if iteration_cnt > self.iteration_max_limit:
                
                self.prompt = self.prompt + 'Answer: ```'
                # if self.method_name is not None:
                #     self.prompt = self.prompt.strip('\n') + f'\n\n{self.method_name}: Answer: ```'
                # else:
                #     self.prompt = self.prompt.strip('\n') + '\n\nAnswer: ```'
                
                original_output = GptCompletion(engine=self.model,
                                        prompt=self.prompt,
                                        max_tokens=128,
                                        temperature=self.temperature,
                                        top_p=1,
                                        frequency_penalty=self.frequency_penalty,
                                        n=1,
                                        stream=False,
                                        # stop='```.',
                                        debug=self.verbose,
                                        )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.gpt_error = f'Max iteration limit hit. limit={self.iteration_max_limit}.'
                self.predicted_result = original_result
                self.prompts.append('\n\n' + self.prompt + original_result)
                self.original_output.append('ForceAnswer: ' + original_result)
                break
            elif answer_type == 'Answer':
                self.predicted_result = answer.split('```')[-1]
                break
            elif answer_type in self.supported_code_types:
                
                renewed_df = self._executor(self.source_table_df, answer, answer_type)
                
                i = len(self.series_dfs) - 1
                while i >= 0 and (renewed_df is None): # or renewed_df.shape[0] == 0):
                    self.source_table_df = self.series_dfs[i]
                    renewed_df = self._executor(self.source_table_df, answer, answer_type)
                    if renewed_df is not None:
                        self.gpt_error = None
                        break
                    i -= 1
                self.source_table_df = renewed_df
                
                if renewed_df is None or answer in self.code_history:
                    # self._gen_gpt_prompt()
                    
                    self.prompt = self.prompt + 'Answer: ```'
                    # if self.method_name is not None:
                    #     self.prompt = self.prompt.strip('\n') + f'\n\n{self.method_name}: Answer: ```'
                    # else:
                    #     self.prompt = self.prompt.strip('\n') + '\n\nAnswer: ```'
                    original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.',
                                            debug=self.verbose,
                                            )
                    original_result = original_output['choices'][0]['text'].replace('\n', '')
                    # self.gpt_error = f'Cannot execute {answer}. Unknown error.'
                    self.predicted_result = original_result
                    self.prompts.append('\n\n' + self.prompt + original_result)
                    self.original_output.append('ForceAnswer: ' + original_result)
                    break   
                
                self.code_history.append(answer)
                data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
                if not maintain_df_ids:
                    intermediate_prompt_template = self.prompt_template_dict['intermediate_prompt_template'][answer_type]
                else:
                    intermediate_prompt_template = self.prompt_template_dict['intermediate_prompt_template'][answer_type].replace(':\n', f" (DF{iteration_cnt}):\n")
                
                if self.method_name is not None:    
                    self.prompt = self.prompt.strip('\n') + original_result + '```.\n\n' + intermediate_prompt_template.format(data_table, self.utterance)
                else:
                    self.prompt = self.prompt.strip('\n') + '\n\n' + original_result + '```.\n\n' + intermediate_prompt_template.format(data_table, self.utterance)
                self.series_dfs.append(renewed_df)
                
                if early_stop_when_table_has_answer and renewed_df.shape[0] == 1 \
                    and renewed_df.columns[0] == 'answer' and len(renewed_df.columns) == 1:
                    self.predicted_result = 'yes' if renewed_df.iloc[0]['answer'] == 1 else 'no'
                    self.original_output.append(f"early stop encountered: {self.predicted_result} with ans={renewed_df.iloc[0]['answer']}")
                    if self.verbose:
                        print(f"early stop when encountered: {renewed_df}")
                        print(f"with answer = {self.predicted_result}")
                    break
                
            else:
                self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
                self.prompt = self.prompt + 'Answer: ```'
                # if self.method_name is not None:
                #     self.prompt = self.prompt.strip('\n') + f'\n\n{self.method_name}: Answer: ```'
                # else:
                #     self.prompt = self.prompt.strip('\n') + '\n\nAnswer: ```''
                original_output = GptCompletion(
                                            engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            debug=self.verbose,
                                            )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.original_output.append('ForceAnswer: ' + original_result)
                self.predicted_result = original_result
                self.prompts.append('\n\n' + self.prompt + original_result)
                break
        self.prompt = self.prompts[-1]
        
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
            'history_prompts': self.prompts,
            'gpt_original_output': self.original_output, 
            'training_demo_ids': self.training_demo_ids
        }
    

class CodexAnswerCOTExecutor_SplitFact_majorityVote(CodexAnswerCOTExecutor_SplitFact):
    def __init__(self, 
                 prompt_template_json, 
                 qid, 
                 utterance, 
                 source_csv, 
                 target_value, 
                 base_path='./', 
                 demo_file=None, 
                 sep=',', 
                 splitfact_template_json=None, 
                 splitfact_demo_file=None,
                 splitfact_model=None,
                 table_title=None,
                 verbose=False,
                 method_name=None
                ):
        super().__init__(
                 prompt_template_json, 
                 qid, 
                 utterance, 
                 source_csv, 
                 target_value, 
                 base_path, 
                 demo_file, 
                 sep, 
                 splitfact_template_json, 
                 splitfact_demo_file,
                 splitfact_model,
                 table_title,
                 verbose,
                 method_name=method_name,
                 )
    
    def _get_gpt_prediction_splitfact(self, 
                                      repeat_num=5,
                                      temperature=0.2,
                                      maintain_df_ids=True, 
                                      split_fact=True, 
                                      seperate_errors=False,
                                      temperature_mixing=True,
                                      vague_prediction_force_temp0=None,
                                      early_stop_when_table_has_answer=True
                                      ):
        if temperature is not None and temperature >= 0 and temperature < 2:
            self.temperature = temperature
        all_original_outputs = []
        all_predictions = []
        all_predictions_with_error = []
        gpt_errors = []
        for _ in range(repeat_num):
            self.gpt_error = None
            self.prompts = []
            self.original_output = []
            self._read_data()
            self._gen_gpt_prompt(maintain_df_ids=maintain_df_ids)
            super()._get_gpt_prediction_splitfact(
                maintain_df_ids=maintain_df_ids, 
                split_fact=split_fact,
                early_stop_when_table_has_answer=early_stop_when_table_has_answer
            )
            
            gpt_errors.append(self.gpt_error)
            if self.gpt_error is not None:
                all_predictions_with_error.append(self.predicted_result)
            else:
                all_predictions.append(self.predicted_result)
            all_original_outputs.append(self.original_output)
        
        if not seperate_errors:
            all_predictions += all_predictions_with_error
        
        if temperature_mixing:
            original_temperature = self.temperature
            self.temperature = 0
            self.prompts = []
            self.original_output = []
            self._read_data()
            self._gen_gpt_prompt(maintain_df_ids=maintain_df_ids)
            super()._get_gpt_prediction_splitfact(
                maintain_df_ids=maintain_df_ids, 
                split_fact=split_fact,
                early_stop_when_table_has_answer=early_stop_when_table_has_answer
            )
            gpt_errors.append(self.gpt_error)
            all_predictions.append(self.predicted_result)
            self.temperature = original_temperature
            all_original_outputs.append(self.original_output)
        
        self.all_predictions = all_predictions
        self.all_predictions_with_error = all_predictions_with_error
        self.original_output = all_original_outputs
        
        from collections import Counter
        if len(all_predictions) > 0:
            counter = Counter(all_predictions)
        else:
            counter = Counter(all_predictions_with_error)

        majority = counter.most_common(1)[0][0]
        if vague_prediction_force_temp0 is not None \
            and temperature_mixing \
            and counter[majority] <= (repeat_num + 1) * vague_prediction_force_temp0:
            self.predicted_result = all_predictions[-1]
            self.gpt_error = gpt_errors
        else:
            self.predicted_result = majority
            self.gpt_error = gpt_errors
    
    def _log_dict(self):
        return {
            'id': self.qid,
            'utterance': self.utterance,
            'source_csv': self.source_csv,
            'target_value': self.target_value,
            'predicted_value': self.predicted_result,
            'all_predictions': self.all_predictions,
            'all_predictions_with_error': self.all_predictions_with_error,
            'gpt_error': self.gpt_error,
            'prompt': self.prompt,
            'execution_match': self.execution_acc,
            'execution_err': self.execution_err,
            'predicted_sql': self.predicted_sql,
            'df_reformat_sql': self.reformat_sql,
            # 'history_prompts': self.prompts,
            'gpt_original_output': self.original_output, 
            'training_demo_ids': self.training_demo_ids
        }
                
if __name__ == '__main__': 
    print(_split_facts("(Fact statement 1) AND (Fact statement 2)"))