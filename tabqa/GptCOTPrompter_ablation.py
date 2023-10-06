from tabqa.GptCOTPrompter import *
from tabqa.GptCOTPrompter_BeamSeach import *

class CodexAnswer_template(CodexAnswerCOTExecutor_template):
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

        super().__init__(
            prompt_template_json, 
            qid, 
            utterance, 
            source_csv, 
            target_value, 
            base_path, 
            demo_file, 
            sep)
    
    def _gen_gpt_prompt(
        self, 
        nearest_neighbor=False, 
        ft=None, 
        maintain_df_ids=False
        ):
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
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        elif not nearest_neighbor and maintain_df_ids:
            for demo in demos[0:self.max_demo]:
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        
        self.prompt = self.demo_prompt + self.prompt + '\n\n'

class CodexAnswer_HighTemperaturMajorityVote(CodexAnswerCOTExecutor_HighTemperaturMajorityVote):
    
    def __init__(
        self, 
        prompt_template_json, 
        qid, 
        utterance, 
        source_csv, 
        target_value, 
        base_path='./', 
        demo_file=None, 
        sep=','
        ):
        
        super().__init__(
            prompt_template_json, 
            qid, 
            utterance, 
            source_csv, 
            target_value, 
            base_path=base_path, 
            demo_file=demo_file, 
            sep=sep
            )
    
    def _gen_gpt_prompt(
        self, 
        nearest_neighbor=False, 
        ft=None, 
        maintain_df_ids=False
        ):
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
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        elif not nearest_neighbor and maintain_df_ids:
            for demo in demos[0:self.max_demo]:
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        
        self.prompt = self.demo_prompt + self.prompt + '\n\n'
            
            
class CodexAnswer_LeverVote(CodexAnswerCOTExecutor_LeverVote):
    def __init__(self, prompt_template_json, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        super().__init__(prompt_template_json, qid, utterance, source_csv, target_value, base_path=base_path, demo_file=demo_file, sep=sep)
        self.temperature = 0.6
        self.line_limit = 10

    def _gen_gpt_prompt(
        self, 
        nearest_neighbor=False, 
        ft=None, 
        maintain_df_ids=False
        ):
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
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        elif not nearest_neighbor and maintain_df_ids:
            for demo in demos[0:self.max_demo]:
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        
        self.prompt = self.demo_prompt + self.prompt + '\n\n'

class CodexAnswer_BeamSeach(CodexAnswerCOTExecutor_BeamSeach):
    def __init__(self, prompt_template_json, qid, utterance, source_csv, target_value, base_path='./', demo_file=None, sep=','):
        super().__init__(prompt_template_json, qid, utterance, source_csv, target_value, base_path=base_path, demo_file=demo_file, sep=sep)
        self.temperature = 0.6
        # tune hyperparameters

    def _gen_gpt_prompt(
        self, 
        nearest_neighbor=False, 
        ft=None, 
        maintain_df_ids=False
        ):
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
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        elif not nearest_neighbor and maintain_df_ids:
            for demo in demos[0:self.max_demo]:
                self.demo_prompt += self.prompt_template.format(demo['tables'][0], demo['utterance']) + '\n\n'
                self.demo_prompt += demo['responses'][-1] + '\n\n'
        
        self.prompt = self.demo_prompt + self.prompt + '\n\n'