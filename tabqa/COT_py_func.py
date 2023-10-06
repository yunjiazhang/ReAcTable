from tabqa.GptPrompter import *

class CodexAnswerCOTExecutor(CodexAnswer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        self.sep = sep
        super().__init__(qid, utterance, source_csv, target_value, base_path)
        self.demo_file = demo_file
        self.model = 'davinci-codex-002-msft'
        assert self.demo_file is not None, "The demo file should not be None for CodexAnswerCOTExecutor."
        if "sql-py" in demo_file:
            self.prompt_template = """The database table DF is shown as follows:
{}

Answer the following question based on the data above: "{}". Execute SQL or Python code step-by-step and finally answer the question. Choose from generating a SQL, Python code, or directly answering the question.
"""
        else:
            self.prompt_template = """The database table DF is shown as follows:
{}

Answer the following question based on the data above: "{}". Execute SQL step-by-step and finally answer the question. Choose from generating a SQL or directly answering the question.
"""
        self.supported_code_types = ['SQL', 'Python']
        self.line_limit = 50
        self.frequency_penalty = 0.7
        self.source_table_df.columns = [normalize_col_name(c) for c in self.source_table_df.columns.tolist()]
        self.series_dfs = [self.source_table_df]
        self.iteration_max_limit = 5
        
    def _read_data(self, ):
        self.source_table_df = pd.read_csv(self.source_csv, on_bad_lines='skip', sep=self.sep)
        # print("Handler: ", self.source_table_df)
        self.source_schema = [normalize_col_name(c) for c in list(self.source_table_df.columns)]
        self.source_table_df.columns = self.source_schema
        self.data_examples = ''
        for i in range(min(100, self.source_table_df.shape[0])):
            self.data_examples += '\t'.join([str(i) for i in self.source_table_df.iloc[i].tolist()]) + '\n'
        
    def _executor(self, df, code, code_type):
        import pandasql as ps
        DF = df
        try:
            if 'SQL' in code_type:
                renewed_df = ps.sqldf(code)
            elif 'Python' in code_type:
                if "try:" not in code and "def" in code and '(s)' in code:
                    ##################################################
                    # there is a function but the function is not safe
                    ##################################################
                    codes = code.split('\n')
                    safe_codes = []
                    add_indent = False
                    for line in codes:
                        if 'def' in line:
                            safe_codes.append(line)
                            safe_codes.append('    try:')
                            add_indent = True
                        elif 'return' in line:
                            safe_codes.append('    ' + line)
                            safe_codes.append('    except:')
                            safe_codes.append('        return s')
                            add_indent = False
                        elif add_indent:
                            safe_codes.append('    ' + line)
                        else:
                            safe_codes.append(line)
                    code = '\n'.join(safe_codes)
                # print(code)
                exec(code, locals(), locals())
                renewed_df = DF
            return renewed_df
        except Exception as e:
            self.gpt_error = f'Cannot execute {code_type} {code} on \n{df.to_string()}\nError: {str(e)}'
            return None
    
    def _gen_gpt_prompt(self):
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
        self.prompt = self.prompt_template.format(data_table, self.utterance)
        with open(os.path.join(self.base_path, self.demo_file), 'r') as f:
            demo = f.read()
            self.prompt = demo + '\n' + self.prompt
            
    def _get_gpt_prediction(self):
        self.original_output = []
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()]        
        
        iteration_cnt = 0
        while True:
            iteration_cnt += 1
            self.prompts.append(self.prompt)
            original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=1024,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```.')
            
            original_result = original_output['choices'][0]['text'].strip('\n')
            answer_type = original_result.split(":")[0]
            answer = original_result.split('```')[-1]
            self.original_output.append(original_result)
            
            if iteration_cnt > self.iteration_max_limit:
                self.prompt += '\nAnswer: ```'
                original_output = openai.Completion.create(engine=self.model,
                                        prompt=self.prompt,
                                        max_tokens=1024,
                                        temperature=0,
                                        top_p=1,
                                        frequency_penalty=self.frequency_penalty,
                                        n=1,
                                        stream=False,
                                        stop='```.')
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
                break
            elif answer_type == 'Answer':
                self.predicted_result = answer.split('```')[-1]
                break
            elif answer_type in self.supported_code_types:
                renewed_df = self._executor(self.source_table_df, answer, answer_type)
                
                i = len(self.series_dfs) - 1
                while i >= 0 and (renewed_df is None or renewed_df.shape[0] == 0):
                    self.source_table_df = self.series_dfs[i]
                    renewed_df = self._executor(self.source_table_df, answer, answer_type)
                    if renewed_df is not None:
                        self.gpt_error = None
                    i -= 1
                self.source_table_df = renewed_df
                
                if renewed_df is None:
                    # self._gen_gpt_prompt()
                    self.prompt += '\nAnswer: ```'
                    original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=1024,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```.')
                    original_result = original_output['choices'][0]['text'].replace('\n', '')
                    self.predicted_result = original_result
                    break   
                    
                data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
                self.prompt = self.prompt + '\n' + original_result + '```.\n\n' + self.prompt_template.format(data_table, self.utterance)
                self.series_dfs.append(renewed_df)
                
            else:
                self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
                self.prompt += '\nAnswer: ```'
                original_output = openai.Completion.create(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=1024,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```.')
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
                break
        self.prompt = self.prompts[-1]
        

class CodexAnswerCOTExecutor_SQA(CodexAnswerCOTExecutor):
    pass