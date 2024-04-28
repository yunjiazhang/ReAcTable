# ReAcTable
This is the code repo for VLDB 2024 paper: ***ReAcTable: Enhancing ReAct for Table Question Answering***

# Organization
- ```dataset/``` contains the example dataset WikiTQ. This dataset is cloned from the [original WikiTQ dataset](https://github.com/ppasupat/WikiTableQuestions).  
- ```notebooks/``` contains the example usage of this github repo. 
- ```tabqa/``` contains the main script of the ReAcTable framework. The major component that is used for ReAcTable TQA is ```CodexAnswerCOTExecutor_HighTemperaturMajorityVote``` class. 
- ```requirement.txt``` contains all the requried packages. Follow the next instructions for package installation.

# Installing the enviornment 
- To begin with, first create an conda enviornment: ```conda create -n reactable python=3.9```
- Then install the reactable package by running ```pip install -e .```


# Usage example
For usage a full usage example, see the notebook for [WikiTQ](./notebooks/ReAcTable-MajorityVote-WikiTQ.ipynb).

- First import all packages needed.
```python
import dotenv
from joblib import Parallel, delayed
from tabqa.GptCOTPrompter_BeamSeach import *
```

- Then, initialize the API key for OpenAI. To do this, you need to have your own ```.env``` file, containing a text line inidcating your API key. For example, ```OPENAI_API_KEY=your_key```.
```python
config = dotenv.dotenv_values("../.env")
openai.api_key = config['OPENAI_API_KEY']
```

- Load WikiTQ dataset.
```python
dataset = pd.read_csv('../dataset/WikiTableQuestions/data/pristine-unseen-tables.tsv', sep='\t')
```

- Setup the running parameters. Keep ```program``` and  ```template``` unchanged. Other parameters are: 1) ```max_demo``` is the number of few shot examples used; 2) gpt_model is the model you use; 3) ```n_threads``` is the number of thread used to evaluate the benchmakr; and 4) ```maxLimit``` is the max number of tests you want to run, let ```maxLimit = float('inf')``` to run all the test queries. 
```python
max_demo = 5
gpt_model = 'gpt-4' # the original results are obtained with Codex, which is deprecated.
program = 'sql-py'
template = 'original-sql-py-no-intermediate'    
n_threads = 3
maxLimit = 5
```

- With the parallel funtion defined as follows, simply run and log the output to a json file. 
```python
def parallel_func(i):
    max_retry = 3
    while max_retry>0:
        try:
            codex_prompter = CodexAnswerCOTExecutor_HighTemperaturMajorityVote(
                                                f'prompt_template/{template}.json',
                                                dataset.iloc[i]['id'], 
                                                dataset.iloc[i]['utterance'], 
                                                dataset.iloc[i]['context'], 
                                                dataset.iloc[i]['targetValue'],  
                                                base_path='../dataset/WikiTableQuestions/',
                                                demo_file=f'few-shot-demo/WikiTQ-{program}.json',
                                                )
            codex_prompter.max_demo = max_demo
            codex_prompter.model = gpt_model
            codex_prompter._gen_gpt_prompt(False)
            codex_prompter._get_gpt_prediction_majority_vote(repeat_times=5)
            log = codex_prompter._log_dict()
            break
        except Exception as e:
            log = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            if "model's maximum context length" in str(e):
                return log
            max_retry -= 1
    return log
output_result_file = f'../dataset/WikiTableQuestions/results/CodexAnswerCOTExecutor_HighTemperaturMajorityVote_{template}_{program}_results_pristine-unseen-tables_limit{maxLimit}_model{gpt_model}.json'
logs = Parallel(
    n_jobs=n_threads, require='sharedmem'
    )(
        delayed(parallel_func)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0])))
    )    
json.dump(logs, open(output_result_file, 'w'), indent=4)
```

- Finally, evaluate the results with the original Python2 evaluator. 
```python
os.system(f'cd ../dataset/WikiTableQuestions/ && python2 evaluator.py ./results/{output_result_file.split("/")[-1]} ')
```