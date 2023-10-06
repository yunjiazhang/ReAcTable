from dateutil import parser
from tabqa.GptCOTPrompter import *
from tabqa.GptConnector import *
import sqlite3
import string
import re

class Codex_PAL(CodexAnswerCOTExecutor_template):
    def __init__(self, prompt_template_json, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        super().__init__(prompt_template_json, qid, utterance, source_csv, target_value, base_path, demo_file, sep)
        self.model = 'mp-atoi-codex'
        self.max_demo = 7
        self.demo_ids = None
    
    def _gen_gpt_prompt(self, nearest_neighbor=False, ft=None):
        
        data_table = table_formater(self.source_table_df, permute_df=False, line_limit=self.line_limit)
        self.prompt = self.prompt_template.format(data_table, self.utterance)
        
        # format demo
        assert '.json' in self.demo_file, "Use json file as the demo file format"
        self.demo_prompt = ""
        demos = json.load(open(os.path.join(self.base_path, self.demo_file)))

        if self.demo_ids is not None:
            demos = [demos[i] for i in range(len(demos)) if i in self.demo_ids]
        
        for demo in demos[0:self.max_demo]:
            for i in range(len(demo['tables'])):
                if i == 0:
                    self.demo_prompt += self.prompt_template.format(demo['tables'][i], demo['utterance']) + '\n\n'
                if 'Answer' in demo['responses'][i]:
                    self.demo_prompt += f"\nThe output after executing the codes above is shown below. Asnwer the question: \"{self.utterance}\".\n\n{demo['tables'][-1]}\n\n{demo['responses'][i]}\n\n"  
                else:
                    self.demo_prompt += f'Step {i+1} - ' + demo['responses'][i] + '\n'
                
        self.prompt = self.demo_prompt + self.prompt + '\n'
    
    def _get_gpt_prediction(self):
        self.original_output = []
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower() for c in self.source_table_df.columns.tolist()]        
        
        original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=256,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='\n\n'
                                           )
        
        self.original_output.append(original_output.choices[0].text.strip('\n'))
        codes = original_output.choices[0].text.strip('\n').split('Step')
        renewed_df = self.source_table_df
        for code in codes:
            match = re.search(r'```(.+?)```', code, re.DOTALL)
            if match and '```' in code:
                extracted_code = match.group(1)
                # print(extracted_code)
                if 'SQL:' in code:
                    renewed_df = self._executor(renewed_df, extracted_code, 'SQL')
                elif 'Python:' in code:
                    renewed_df = self._executor(renewed_df, extracted_code, 'Python')
        
        new_table_text  = table_formater(renewed_df, permute_df=False, line_limit=self.line_limit)
        self.prompt += f"\nThe output after executing the codes above is shown below. Asnwer the question: \"{self.utterance}\".\n\n{new_table_text}\n\nAnswer: ```"
        original_output = GptCompletion(engine=self.model,
                                            prompt=self.prompt,
                                            max_tokens=256,
                                            temperature=self.temperature,
                                            top_p=1,
                                            frequency_penalty=self.frequency_penalty,
                                            n=1,
                                            stream=False,
                                            stop='```'
                                           )
        self.predicted_result = original_output['choices'][0]['text']
        self.prompt += self.predicted_result
        
        
        
