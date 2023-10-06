from dateutil import parser
from tabqa.GptPrompter import *
from tabqa.GptConnector import *
import sqlite3
import string
import re
from dateutil import parser
import datetime

PYTHON_PREPARE = """
import pandas as pd
import numpy as np
"""

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(JSONEncoder, self).default(obj)

def contains_sqlite_functions(sql):
    sqlite_functions = [
        "SUBSTR", "SUBSTRING", "TRIM", "REPLACE", "ROUND", 
        "ABS", "LENGTH", "LOWER", "UPPER", "JULIANDAY", 
        "STRFTIME", "strftime", "CAST", "COALESCE", "IFNULL", 
        "NULLIF", "GROUP_CONCAT", "SUBSTRING_INDEX",
        # "DATE", "TIME", "DATETIME",
    ]
    pattern = "|".join(sqlite_functions)
    return bool(re.search(pattern, sql))

def normalize_date(date_string):
    DEFAULT_DATE = datetime.datetime(1900, 1, 1)
    try:
        parsed_date = parser.parse(date_string, default=DEFAULT_DATE)
        if parsed_date is None \
            or not all([parsed_date.year, parsed_date.month, parsed_date.day]):
            return date_string
        normalized_date = datetime.datetime.strftime(parsed_date, "%Y-%m-%d")
        if '1900' in str(normalized_date) and '1900' not in date_string:
            return date_string
        else:
            return normalized_date
    except Exception as e:
        return date_string

def normalize_sql(input_string):
    # Replace en dash, em dash, and other types of dashes with a standard hyphen
    output_string = input_string.replace("–", "-").replace("—", "-").replace("―", "-").replace("−", "-")
    return output_string

def normalize_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(',', '').astype(float)
                continue
            except:
                pass
            
            try:
                df[col] = df[col].apply(lambda x: eval(x.split('=')[1]))
                continue
            except:
                pass
            
    return df

def normalize_date_value(df):
    for col in df.columns:
        if df[col].dtype == object and 'date' in col:
            try:
                df[col] = df[col].apply(lambda x: normalize_date(x))
                continue
            except:
                pass
    return df

def normalize_sep_value(df):
    def normalize_sep(s):
        if type(s) == str:
            return s.replace('|', ' ').strip(' ')
        else:
            return s
    for col in df.columns:
        df[col] = df[col].apply(lambda x: normalize_sep(x))
    return df

def normalize_null_value(df):
    def normalize_null(s):
        if type(s) == str and s.replace(' ', '').lower() in [
            'n.a', 'n/a', 'n.a.', 'n-a', 'nan', 'none', 'null'
        ]:
            return None
        else:
            return s
    for col in df.columns:
        df[col] = df.apply(lambda x: normalize_null(x[col]), axis=1)
    return df

def normalize_data_frame(df):
    df = normalize_sep_value(df)
    df = normalize_numeric_columns(df)
    df = normalize_null_value(df)
    df = normalize_date_value(df)
    return df

def to_safe_python_code(code):
    codes = code.split('\n')
    safe_codes = []
    add_indent = False
    for line in codes:
        if add_indent and not line.startswith('    '):
            safe_codes.append('    except:')
            safe_codes.append('        return 0')
            add_indent = False
        if 'def' in line:
            safe_codes.append(line)
            safe_codes.append('    try:')
            add_indent = True
        elif add_indent:
            safe_codes.append('    ' + line)
        else:
            safe_codes.append(line)
    code = '\n'.join(safe_codes)
    return code

class CodexAnswerCOTExecutor(CodexAnswer):
    def __init__(self, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=',', force_answer_when_empty=None, iteration_max_limit=5):
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
        self.line_limit = 20
        self.frequency_penalty = 0
        self.source_table_df.columns = [normalize_col_name(c) for c in self.source_table_df.columns.tolist()]
        self.series_dfs = [self.source_table_df]
        self.iteration_max_limit = iteration_max_limit
        self.force_answer_when_empty = force_answer_when_empty
        self.original_output = []
        
    def _read_data(self, normalize_df=True):
        self.source_table_df = pd.read_csv(os.path.join(self.base_path, self.source_csv), on_bad_lines='skip', sep=self.sep)
        # # print("Handler: ", self.source_table_df)
        self.source_schema = [normalize_col_name(c) for c in list(self.source_table_df.columns)]
        self.source_table_df.columns = self.source_schema
        self.data_examples = ''
        for i in range(min(100, self.source_table_df.shape[0])):
            self.data_examples += '\t'.join([str(i) for i in self.source_table_df.iloc[i].tolist()]) + '\n'
        self.tmp_db_name = './tmp/' + ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=20)) + '.db'
        if normalize_df:
            self.source_table_df = normalize_data_frame(self.source_table_df)
        self.series_dfs = [self.source_table_df]
        
    def _executor(self, df, code, code_type, safe=True):
        import pandasql as ps
        import re
        DF = df
        # ======================================
        # maintain the history of dataframes
        for i, hist_df in enumerate(self.series_dfs):
            locals()[f"DF{i}"] = hist_df
        # ======================================
        try:
            if 'SQL' in code_type:
                if contains_sqlite_functions(code):
                    # print(f"Connecting to SQLite for execution code: {code}.")
                    conn = sqlite3.connect(self.tmp_db_name)
                    df.to_sql('DF', conn, if_exists='replace', index=False)
                    renewed_df = pd.read_sql(normalize_sql(code), conn)
                    conn.close()
                    os.system(f"rm {self.tmp_db_name}")
                else:
                    renewed_df = ps.sqldf(normalize_sql(code))
            elif 'Python' in code_type:
                if safe and "try:" not in code and "def" in code and '(s)' in code:
                    ##################################################
                    # there is a function but the function is not safe
                    ##################################################
                    code = to_safe_python_code(code)
                    
                # print("Code = ", code)
                # print(DF)
                exec(PYTHON_PREPARE + code, locals(), locals())
                renewed_df = DF
                # print(DF)
            return renewed_df
        except Exception as e:
            self.gpt_error = f'Cannot execute {code_type} {code} on \n{df.to_string()}\nError: {str(e)}'
            # print(str(e))
            if contains_sqlite_functions(code) and 'SQL' in code_type:
                os.system(f"rm {self.tmp_db_name}")
            # exit(1)
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
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()]        
        
        iteration_cnt = 0
        while True:
            iteration_cnt += 1
            self.prompts.append(self.prompt)
            
            if iteration_cnt >= self.iteration_max_limit:
                self.prompt += '\nAnswer: ```'
                original_output = GptCompletion(engine=self.model,
                                        prompt=self.prompt,
                                        max_tokens=128,
                                        temperature=0,
                                        top_p=1,
                                        frequency_penalty=self.frequency_penalty,
                                        n=1,
                                        stream=False,
                                        # stop='```.'
                                        )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
                self.original_output.append('\nAnswer: ```' + original_result)
                self.prompt += original_result
                self.prompts.append(self.prompt)
                break
            
            
            original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                           )
            
            original_result = original_output['choices'][0]['text'].strip('\n')
            answer_type = original_result.split(":")[0]
            answer = original_result.split('```')[-1]
            self.original_output.append(original_result)
            
            if answer_type == 'Answer':
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
                    original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                    original_result = original_output['choices'][0]['text'].replace('\n', '')
                    self.predicted_result = original_result
                    break   
                    
                data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
                self.prompt = self.prompt + '\n' + original_result + '```.\n\n' + self.prompt_template.format(data_table, self.utterance)
                self.series_dfs.append(renewed_df)
                
            else:
                self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
                self.prompt += '\nAnswer: ```'
                original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=0,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
                break
        
        self.prompt = self.prompts[-1] + self.predicted_result

        # safe execution
        if self.force_answer_when_empty is not None:
            if self.source_table_df is None or self.source_table_df.shape[0] == 0:
                self.predicted_result = self.force_answer_when_empty


class CodexAnswerCOTExecutor_template(CodexAnswerCOTExecutor):
    def __init__(
        self, 
        prompt_template_json, 
        qid, 
        utterance, 
        source_csv, 
        target_value, 
        base_path='./', 
        demo_file=None, 
        sep=','):
        super().__init__(qid, utterance, source_csv, target_value, base_path, demo_file, sep)
        self.prompt_template_dict = json.load(open(os.path.join(self.base_path, prompt_template_json)))
        self.prompt_template = self.prompt_template_dict['prompt_template']
        self.max_demo = 5
        self.line_limit = 10
        self.use_data_instance_to_calculat_similarity = True
        self.temperature = 0
        self.demo_ids = None
        self.code_history = []
        self.original_output = []

    def _gen_gpt_prompt(self, nearest_neighbor=False, ft=None, maintain_df_ids=False):
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
        self.prompt = self.prompt_template.format(data_table, self.utterance)
        if maintain_df_ids:
            self.prompt = self.prompt.replace("DF", "DF0")
        
        # format demo
        assert '.json' in self.demo_file, "Use json file as the demo file format"
        self.demo_prompt = ""
        demos = json.load(open(os.path.join(self.base_path, self.demo_file)))
        
        if self.demo_ids is not None:
            demos = [demos[i] for i in self.demo_ids]

        if nearest_neighbor is False and not maintain_df_ids:
            for demo in demos[0:self.max_demo]:
                for i in range(len(demo['tables'])):
                    if i == 0:
                        self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']) + '\n\n'
                    else:
                        if 'SQL:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['SQL'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                        elif 'Python:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['Python'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                    self.demo_prompt += demo['responses'][i] + '\n\n'
        elif not nearest_neighbor and maintain_df_ids:
            for demo in demos[0:self.max_demo]:
                for i in range(len(demo['tables'])):
                    if i == 0:
                        self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']).replace("DF", "DF0") + '\n\n'
                    else:
                        if 'SQL:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['SQL'].replace(":\n", f" (DF{i}):\n").format(demo['tables'][i], demo['utterance']) + '\n\n'
                        elif 'Python:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['Python'].replace(":\n", f" (DF{i}):\n").format(demo['tables'][i], demo['utterance']) + '\n\n'
                    self.demo_prompt += demo['responses'][i] + '\n\n'
            
            
        elif nearest_neighbor and not self.use_data_instance_to_calculat_similarity:
            # utterance_embedding = get_utterance_embedding(data_table.split('[HEAD]: ')[0].split(''), ft)
            utterance_embedding = get_utterance_embedding(self.utterance, ft)
            similarities = []
            for demo in demos:
                # similarities.append(-1 * get_embedding_cos_sim(utterance_embedding, get_utterance_embedding(demo['tables'][0].split('\n')[0], ft)))
                similarities.append(-1 * get_embedding_cos_sim(utterance_embedding, get_utterance_embedding(demo['utterance'], ft)))
                
            demo_index = np.argsort(similarities)
            # # print(similarities)
            self.training_demo_ids = [int(i) for i in list(demo_index[0:self.max_demo])]
            for idx in demo_index[0:self.max_demo]:
                demo = demos[idx]
                for i in range(len(demo['tables'])):
                    if i == 0:
                        self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']) + '\n\n'
                    else:
                        if 'SQL:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['SQL'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                        elif 'Python:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['Python'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                    self.demo_prompt += demo['responses'][i] + '\n\n'
        
        elif nearest_neighbor and self.use_data_instance_to_calculat_similarity:
            utterance_embedding = get_utterance_embedding('\n'.join(data_table.split('---\n')[1].split('\n')[0:2]), ft)

            similarities = []
            for demo in demos:
                similarities.append(-1 * get_embedding_cos_sim(utterance_embedding, get_utterance_embedding('\n'.join(demo['tables'][0].split('---\n')[1].split('\n')[0:2]), ft)))
                
            demo_index = np.argsort(similarities)
            # # print(similarities)
            self.training_demo_ids = [int(i) for i in list(demo_index[0:self.max_demo])]
            for idx in demo_index[0:self.max_demo]:
                demo = demos[idx]
                for i in range(len(demo['tables'])):
                    if i == 0:
                        self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']) + '\n\n'
                    else:
                        if 'SQL:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['SQL'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                        elif 'Python:' in demo['responses'][i-1]:
                            self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['Python'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                    self.demo_prompt += demo['responses'][i] + '\n\n'
        
        
        self.prompt = self.demo_prompt + self.prompt + '\n'
    
    def _get_gpt_prediction(self, maintain_df_ids=False):
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()]        
        self.code_history = []
        iteration_cnt = 0

        while True:
            iteration_cnt += 1
            self.prompts.append(self.prompt)
            original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                           )
            
            original_result = original_output['choices'][0]['text'].strip('\n')
            answer_type = original_result.split(":")[0]
            answer = original_result.split('```')[-1]
            self.original_output.append(original_result)
            
            if iteration_cnt > self.iteration_max_limit:
                self.prompt += '\nAnswer: ```'
                original_output = GptCompletion(engine=self.model,
                                        prompt=self.prompt,
                                        max_tokens=128,
                                        temperature=self.temperature,
                                        top_p=1,
                                        frequency_penalty=self.frequency_penalty,
                                        n=1,
                                        stream=False,
                                        # stop='```.'
                                        )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
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
                    i -= 1
                self.source_table_df = renewed_df
                
                if renewed_df is None or answer in self.code_history:
                    # self._gen_gpt_prompt()
                    self.prompt += '\nAnswer: ```'
                    original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                    original_result = original_output['choices'][0]['text'].replace('\n', '')
                    self.predicted_result = original_result
                    break   
                
                self.code_history.append(answer)
                data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
                if not maintain_df_ids:
                    intermediate_prompt_template = self.prompt_template_dict['intermediate_prompt_template'][answer_type]
                else:
                    intermediate_prompt_template = self.prompt_template_dict['intermediate_prompt_template'][answer_type].replace(':\n', f" (DF{iteration_cnt}):\n")
                    
                self.prompt = self.prompt.strip('\n') + '\n\n' + original_result + '```.\n\n' + intermediate_prompt_template.format(data_table, self.utterance)
                self.series_dfs.append(renewed_df)
                
            else:
                self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
                self.prompt += '\nAnswer: ```'
                original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
                break
        self.prompt = self.prompts[-1]

        
class CodexAnswerCOTExecutor_decisionAction(CodexAnswerCOTExecutor_template):
    def __init__(self, prompt_template_json, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        super().__init__(prompt_template_json, qid, utterance, source_csv, target_value, base_path=base_path, demo_file=demo_file, sep=sep)
        self.temperature = 0
        self.iteration_max_limit = 5
        self.intermediate_prompt_full_context = True
        self.line_limit = 10
        self.demo_ids = None
        self.original_output = []
        
    def _gen_gpt_prompt(self, nearest_neighbor=False, ft=None):
        ##############################################################
        # data_table = '\t'.join(self.source_schema) + '\n' + self.data_examples
        ##############################################################
        data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
        self.original_prompt = self.prompt_template.format(data_table, self.utterance)
        
        # format demo
        assert '.json' in self.demo_file, "Use json file as the demo file format"
        self.demo_prompts_by_decision = {
            'SQL': [[] for _ in range(10)],
            'Python': [],
            'Answer': []
        }
        
        demos = json.load(open(os.path.join(self.base_path, self.demo_file)))
        for demo in demos:
            utterance = demo['utterance']
            i = 0
            for table, response in zip(demo['tables'], demo['responses']):
                if 'SQL' in response:
                    temp_prompt = self.prompt_template_dict['intermediate_prompt_template']['SQL'].format(table, utterance) + '\n\n' + response
                    if i == 0:
                        self.demo_prompts_by_decision['SQL'][0].append(temp_prompt)
                    else:
                        self.demo_prompts_by_decision['SQL'][1].append(temp_prompt)
                elif 'Answer' in response:
                    temp_prompt = self.prompt_template_dict['intermediate_prompt_template']['SQL'].format(table, utterance) + '\n\n' + response
                    self.demo_prompts_by_decision['Answer'].append(temp_prompt)
                elif 'Python' in response:
                    temp_prompt = self.prompt_template_dict['intermediate_prompt_template']['Python'].format(table, utterance) + '\n\n' + response
                    self.demo_prompts_by_decision['Python'].append(temp_prompt)
                i += 1

        self.demo_prompts_by_decision['SQL'][0] = '\n\n'.join(self.demo_prompts_by_decision['SQL'][0][0:self.max_demo])
        self.demo_prompts_by_decision['SQL'][1] = '\n\n'.join(self.demo_prompts_by_decision['SQL'][1][0:self.max_demo])
        self.demo_prompts_by_decision['Python'] = '\n\n'.join(self.demo_prompts_by_decision['Python'][0:self.max_demo])
        self.demo_prompts_by_decision['Answer'] = '\n\n'.join(self.demo_prompts_by_decision['Answer'][0:self.max_demo])
        
        if os.path.isfile(os.path.join(self.base_path, self.demo_file.replace('.json', '-DA-py-examples.json'))):
            py_demos = json.load(open(os.path.join(self.base_path, self.demo_file.replace('.json', '-DA-py-examples.json'))))
            self.demo_prompts_by_decision['Python'] = []
            for demo in py_demos:
                utterance = demo['utterance']
                for table, response in zip(demo['tables'], demo['responses']):
                    temp_prompt = self.prompt_template_dict['intermediate_prompt_template']['Python'].format(table, utterance) + '\n\n' + response
                    self.demo_prompts_by_decision['Python'].append(temp_prompt)
            self.demo_prompts_by_decision['Python'] = '\n\n'.join(self.demo_prompts_by_decision['Python'][0:self.max_demo])       
        
        self.demo_prompt = ''
        if self.demo_ids is None:
            demos = demos[0:self.max_demo]
        else:
            demos = demos[self.demo_ids]
            
        for demo in demos:
            for i in range(len(demo['tables'])):
                if i == 0:
                    self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']) + '\n\n'
                else:
                    if 'SQL:' in demo['responses'][i-1]:
                        self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['SQL'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                    elif 'Python:' in demo['responses'][i-1]:
                        self.demo_prompt += self.prompt_template_dict['intermediate_prompt_template']['Python'].format(demo['tables'][i], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][i] + '\n\n'
        self.prompt = self.demo_prompt + self.original_prompt + '\n'
    
    def _get_gpt_prediction(self):
        
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()]        
        self.action_prompts = []
        self.decision_prompts = []
        self.decisions = []
        iteration_cnt = 0
        while True:
            # # print("Start iteration")
            
            iteration_cnt += 1
            if iteration_cnt > self.iteration_max_limit:
                break
            self.prompt = self.prompt.strip('\n') + '\n\n'
            self.prompts.append(self.prompt)
            self.decision_prompts.append(self.prompt)
            
            # # print(f"```{self.prompt}```")
            original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=20,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop=':')
            original_result = original_output['choices'][0]['text']
            decision = original_result.strip('\n')
            # original_output = GptCompletion(engine=self.model,
            #                                 prompt=self.prompt,
            #                                 max_tokens=20,
            #                                 temperature=self.temperature,
            #                                 top_p=1,
            #                                 frequency_penalty=self.frequency_penalty,
            #                                 n=1,
            #                                 stream=False,
            #                                 stop='```.')
            # original_result = original_output['choices'][0]['text']
            # decision = original_result.strip('\n').split(':')[0]
            self.decisions.append(decision)
            # # print("original prompt: \n\n")
            # # print(f"'{self.prompt}'")
            # # print(f"'{original_result}'")
            # # print(decision)
            
            if decision in self.supported_code_types:
                data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
                if iteration_cnt == 1 and decision == 'SQL':
                    if self.intermediate_prompt_full_context:
                        temp_prompt = self.demo_prompts_by_decision[decision][0] + '\n\n' + \
                            self.original_prompt.strip('\n') \
                            + '\n\n' + decision + ': ```'
                    else:
                        temp_prompt = self.demo_prompts_by_decision[decision][0] + '\n\n' + \
                            self.prompt_template_dict['intermediate_prompt_template'][decision].format(data_table, self.utterance).strip('\n') \
                            + '\n\n' + decision + ': ```'
                    
                elif decision == 'SQL':
                    if self.intermediate_prompt_full_context:
                        temp_prompt = self.demo_prompts_by_decision[decision][1] + '\n\n' + \
                            self.original_prompt.strip('\n') \
                            + '\n\n' + decision + ': ```'
                    else:
                        temp_prompt = self.demo_prompts_by_decision[decision][1] + '\n\n' + \
                            self.prompt_template_dict['intermediate_prompt_template'][decision].format(data_table, self.utterance).strip('\n') \
                            + '\n\n' + decision + ': ```'
                else:
                    if self.intermediate_prompt_full_context:
                        temp_prompt = self.demo_prompts_by_decision[decision] + '\n\n' + \
                            self.original_prompt.strip('\n') \
                            + '\n\n' + decision + ': ```'
                    else:
                        temp_prompt = self.demo_prompts_by_decision[decision] + '\n\n' + \
                            self.prompt_template_dict['intermediate_prompt_template'][decision].format(data_table, self.utterance).strip('\n') \
                            + '\n\n' + decision + ': ```'
                self.action_prompts.append(temp_prompt)
                original_output = GptCompletion(engine=self.model,
                                            prompt=temp_prompt.strip('\n'),
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                answer = original_output['choices'][0]['text']
                self.original_output.append(decision + ": ```" + answer + "```.")
                # # print(self.original_output[-1])
                # print(f"Executing {answer}")
                # print(self.source_table_df)
                renewed_df = self._executor(self.source_table_df, answer, decision)
                # ===========================================================================
                i = len(self.series_dfs) - 1
                while i >= 0 and (renewed_df is None): # or renewed_df.shape[0] == 0):
                    self.source_table_df = self.series_dfs[i]
                    renewed_df = self._executor(self.source_table_df, answer, decision)
                    if renewed_df is not None:
                        self.gpt_error = None
                    i -= 1
                self.source_table_df = renewed_df
                
                if renewed_df is None:
                    # self._gen_gpt_prompt()
                    self.prompt = self.prompt.strip('\n') + '\n\nAnswer: ```'
                    original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                    original_result = original_output['choices'][0]['text'].replace('\n', '')
                    self.predicted_result = original_result
                    break 
                self.series_dfs.append(renewed_df)
                # ===========================================================================
                
                data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
                self.prompt = self.prompt.strip('\n') + f'\n\n{decision}: ```{answer}' \
                    + '```.\n\n' + self.prompt_template_dict['intermediate_prompt_template'][decision].format(data_table, self.utterance) + '\n\n'
                self.original_prompt = self.original_prompt.strip('\n') + f'\n\n{decision}: ```{answer}' \
                    + '```.\n\n' + self.prompt_template_dict['intermediate_prompt_template'][decision].format(data_table, self.utterance)
            elif decision == 'Answer':
                original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt.strip('\n') + '\n\nAnswer: ```',
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                answer = original_output['choices'][0]['text']
                self.original_output.append(decision + ": ```" + answer + "```.")
                # # print(decision + ": ```" + answer + "```.")
                self.predicted_result = answer
                self.prompt = self.prompt.strip('\n') + f'\n\n{decision}: ```{answer}' + '```.\n\n'
                break
            else:
                self.gpt_error = f'Unsupported code type generated: {decision}'
                self.prompt = self.prompt.strip('\n') + '\n\nAnswer: ```'
                original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=128,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            # stop='```.'
                                            )
                original_result = original_output['choices'][0]['text'].replace('\n', '')
                self.predicted_result = original_result
            
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
            'training_demo_ids': self.training_demo_ids
        }
            
#             if iteration_cnt > self.iteration_max_limit:
#                 self.prompt += '\nAnswer: ```'
#                 original_output = GptCompletion(engine=self.model,
#                                         prompt=self.prompt,
#                                         max_tokens=128,
#                                         temperature=self.temperature,
#                                         top_p=1,
#                                         frequency_penalty=self.frequency_penalty,
#                                         n=1,
#                                         stream=False,
#                                         stop='```.')
#                 original_result = original_output['choices'][0]['text'].replace('\n', '')
#                 self.predicted_result = original_result
#                 break
#             elif answer_type == 'Answer':
#                 self.predicted_result = answer.split('```')[-1]
#                 break
#             elif answer_type in self.supported_code_types:
#                 # # print("df: ", self.source_table_df)
#                 # # print("Code: ", answer)
#                 renewed_df = self._executor(self.source_table_df, answer, answer_type)
#                 # # print("New df: \n", renewed_df)
#                 # # print(renewed_df.columns)
                
#                 i = len(self.series_dfs) - 1
#                 while i >= 0 and (renewed_df is None): # or renewed_df.shape[0] == 0):
#                     self.source_table_df = self.series_dfs[i]
#                     renewed_df = self._executor(self.source_table_df, answer, answer_type)
#                     if renewed_df is not None:
#                         self.gpt_error = None
#                     i -= 1
#                 self.source_table_df = renewed_df
                
#                 if renewed_df is None:
#                     # self._gen_gpt_prompt()
#                     self.prompt += '\nAnswer: ```'
#                     original_output = GptCompletion(engine=self.model,
#                                             prompt=self.prompt,
#                                             max_tokens=128,
#                                             temperature=0,
#                                             top_p=1,
#                                             frequency_penalty=self.frequency_penalty,
#                                             n=1,
#                                             stream=False,
#                                             stop='```.')
#                     original_result = original_output['choices'][0]['text'].replace('\n', '')
#                     self.predicted_result = original_result
#                     break   
                    
#                 data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
#                 intermediate_prompt_template = self.prompt_template_dict['intermediate_prompt_template'][answer_type]
#                 self.prompt = self.prompt + '\n' + original_result + '```.\n\n' + intermediate_prompt_template.format(data_table, self.utterance)
#                 self.series_dfs.append(renewed_df)
                
#             else:
#                 self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
#                 self.prompt += '\nAnswer: ```'
#                 original_output = GptCompletion(engine=self.model,
#                                             prompt=self.prompt,
#                                             max_tokens=128,
#                                             temperature=0,
#                                             top_p=1,
#                                             frequency_penalty=self.frequency_penalty,
#                                             n=1,
#                                             stream=False,
#                                             stop='```.')
#                 original_result = original_output['choices'][0]['text'].replace('\n', '')
#                 self.predicted_result = original_result
#                 break
#         self.prompt = self.prompts[-1]

        
