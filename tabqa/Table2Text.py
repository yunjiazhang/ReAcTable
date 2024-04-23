from tabqa.GptPrompter import *

class Table2Text:

    def __init__(self, ):
        # model = "davinci-codex-002-msft"
        self.model = "text-davinci-002"        
        self.demo = None

    def loadDemo(self,):
        pass

    def generateText(self, df):
        table_text = table_formater(df)
        prompt = f"""
Convert the following table into multiple text descriptions. Generate comprehensive, self-containing descriptions of the table.

The table is:
{table_text}

The text descriptions of the table are: ```
"""

        if self.demo is not None:
            prompt = self.demo + prompt 

        nl_text = openai.Completion.create(engine=self.model,
                                prompt=prompt,
                                max_tokens=4000 - len(table_text),
                                temperature=0,
                                top_p=1,
                                frequency_penalty=0,
                                n=1,
                                stream=False,
                                stop='```').choices[0].text
        return table_text, nl_text


def gen_text(i):
    try:
        context_df = pd.read_csv(f'./dataset/WikiTableQuestions/{dataset.iloc[i]["context"]}', sep=',', on_bad_lines='warn')
        table_text, nl_text = generator.generateText(context_df)
        return {'id': dataset.iloc[i]["id"], 'table_text': table_text, 'nl_text': nl_text, 'error': None}
    except Exception as e: 
        return {'id': dataset.iloc[i]["id"], 'table_text': None, 'nl_text': None, 'error': str(e)}


if __name__ == "__main__":
    dataset = pd.read_csv('./dataset/WikiTableQuestions/data/pristine-unseen-tables.tsv', sep='\t', on_bad_lines='skip')
    generator = Table2Text()
    n_threads = 20
    maxLimit = float('inf')
    # maxLimit = 40
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_threads, require='sharedmem')(delayed(gen_text)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0]))))
    json.dump(res, open('./Table2Text_prestine-unseen-tables.json', 'w'))

    dataset = pd.read_csv('./dataset/WikiTableQuestions/data/training.tsv', sep='\t', on_bad_lines='skip')
    generator = Table2Text()
    n_threads = 20
    maxLimit = float('inf')
    # maxLimit = 40
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_threads, require='sharedmem')(delayed(gen_text)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0]))))
    json.dump(res, open('./Table2Text_training.json', 'w'))