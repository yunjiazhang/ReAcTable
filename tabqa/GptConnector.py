import os
import openai
import multiprocessing
import time
import pprint

ROLES = ["user", "assistant"]

def visualize_messages(messages, show=True):
    pprint.pprint(messages)
    msg = ''
    for message in messages:
        msg += message['role'].upper() + '\n' + message['content'] + '\n'
    if show:
        print("the entire msg: \n" + msg)
    return msg

def prompt2messages(prompt):
    # print(prompt + '\n====================================')
    
    def is_gpt_respones(s):
        return s.startswith("SQL:") \
            or s.startswith("Python:")\
            or s.startswith("Answer:") 
    
    user_role = 'user'
    gpt_role = 'assistant'
    messages = [
        {
            'role': user_role,
            'content': ''
        }
    ]
    lines = prompt.split('\n')
    is_gpt_response = False
    
    for line in lines:
        if is_gpt_response:
            if '```' not in line:
                messages[-1]['content'] += '\n' + line
            else:
                messages[-1]['content'] += '\n' + line
                messages.append(
                    {
                        'role': user_role,
                        'content': ''
                    }
                )
                is_gpt_response = False
        elif is_gpt_respones(line):
            messages.append(
                {
                    'role': gpt_role,
                    'content': line
                }
            )
            if line.count("```") == 1:
                is_gpt_response = True
                
            else:
                is_gpt_response = False
                messages.append(
                    {
                        'role': user_role,
                        'content': ''
                    }
                )
        else:        
            messages[-1]['content'] += '\n' + line

    for i, d in enumerate(messages):
        messages[i] = {
            'role': d['role'],
            'content': d['content'].strip('\n')
        }
        if i == len(messages) - 1 and len(d['content'].strip('\n')) == 0:
            messages.pop(-1)
    # visualize_messages(messages, False)
    return messages

def is_chat_model(model):
    if model in [
        'gpt-4', 
        'gpt-4-0314', 
        'gpt-4-32k', 
        'gpt-4-32k-0314', 
        'gpt-3.5-turbo', 
        'gpt-3.5-turbo-0301',
        'gpt-35-turbo']:
        return True
    else:
        return False
    
def GptCompletion(
    engine, 
    prompt, 
    suffix=None, 
    max_tokens=128, 
    temperature=0,
    top_p=1, 
    n=1,
    stream=False,
    logprobs=None,
    stop=['```.', '``` '],
    presence_penalty=0,
    frequency_penalty=0,
    best_of=1,
    debug=False,
    prompt_end='\n\n'
):
    # if debug:
    #     print("===================================\n\"", prompt, '\"')
    #     print("===================================\n")
        
    def gpt(queue):
        try:
            if not is_chat_model(engine):
                current_prompt = prompt.strip(' ')
                
                while prompt_end is not None \
                        and not current_prompt.endswith(prompt_end) \
                        and not current_prompt.endswith("```") \
                        and not current_prompt.endswith(":"):
                    current_prompt += prompt_end[0]
                
                if debug:
                    print("===================================\n\"" + current_prompt + '\"')
                    print("===================================\n")
                
                output = openai.Completion.create(    
                    engine = engine, 
                    prompt = current_prompt, 
                    suffix=suffix, 
                    max_tokens=max_tokens, 
                    temperature=temperature,
                    top_p=top_p, 
                    n=n,
                    stream=stream,
                    logprobs=logprobs,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    best_of=best_of,
                )
            else:
                messages = prompt2messages(prompt)
                # pprint.pprint(messages)
                output = openai.ChatCompletion.create(    
                    engine = engine, 
                    messages = messages, 
                    # suffix=suffix, 
                    max_tokens=128, 
                    temperature=temperature,
                    top_p=top_p, 
                    n=n,
                    stream=stream,
                    # logprobs=logprobs,
                    stop=stop,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    # best_of=best_of,
                )
                output['choices'][0]['text'] = output['choices'][0]['message']['content'].strip('```')
                if 'Answer:' in output['choices'][0]['text'] and 'Answer: ```' not in output['choices'][0]['text']:
                    output['choices'][0]['text'] = output['choices'][0]['text'].replace('Answer:', 'Answer: ```')
                # print(output)
        except Exception as e:
            print("Error: " + str(e))
            if "rate" in str(e).lower():
                queue.put("Rate limit")
            else:
                queue.put(None)
            return None
        queue.put(output)

    prompt=prompt.strip(' ')
    max_retry = 3
    timeout = 20
    connection_cnt = 0
    while connection_cnt < max_retry:
        connection_cnt += 1
        q = multiprocessing.Queue()
        try:
            p = multiprocessing.Process(target=gpt, args=(q,))
            p.start()
            p.join(timeout)
        except Exception as e:
            if "Rate" in str(e):
                time.sleep(10) 
                continue
            else:
                raise e
        if p.is_alive():
            p.terminate()
            if debug:
                print(f"Function timed out, retrying ({connection_cnt}/{max_retry})...")
            time.sleep(1) 
        else:
            result = q.get()
            if debug:
                print(result)
                print("===================================")
            if result is None:
                raise ValueError("Error encountered.")
            elif result == "Rate limit":
                print("Rate limit detacted. Retrying after 20sec.")
                time.sleep(20) 
                continue
            else:
                return result
    # assert False, f"Gpt connection timeout after {max_retry} retry."
    raise ValueError(f"Gpt connection timeout after {max_retry} retry.")
    
    
        
if __name__ == '__main__':
    import dotenv
    config = dotenv.dotenv_values(".env")
    openai.api_key = config['OPENAI_API_KEY']
    print("Start prompting")
    prompt ="""The database table DF is shown as follows:
[HEAD]: name|c_1989|c_1990|c_1991|c_1992|c_1993|c_1994|c_1995|c_1996|c_1997|c_1998|c_1999|c_2000|c_2001|c_2002|c_2003|c_2004|c_2005|c_2006|c_2007|c_2008|c_2009|c_2010|career_sr|career_win_loss
---
[ROW] 1: Australian Open|A|A|1R|A|2R|3R|2R|1R|A|3R|4R|1R|2R|1R|3R|2R|1R|QF|3R|2R|3R|1R|0 / 18|22–18
[ROW] 2: French Open|1R|2R|4R|1R|1R|3R|1R|A|1R|3R|1R|2R|4R|2R|2R|3R|1R|1R|1R|2R|1R|A|0 / 20|17–20
[ROW] 3: Wimbledon|A|1R|A|A|A|A|1R|A|1R|A|2R|2R|3R|2R|2R|2R|2R|2R|2R|1R|2R|A|0 / 14|11–14
...
[ROW] 17: Annual Win-Loss|nan|2–4|7–5|3–5|6–4|2–1|5–4|2–1|12–6|10–9|10–7|12–9|13–9|9–9|2–7|8–5|7–7|3–8|4–3|2–3|1–2|0–0|nan|120–108
[ROW] 18: Year End Ranking|235|62|43|43|55|46|102|118|29|41|34|31|22|35|62|52|58|52|37|52|68|–|nan|nan

Answer the following question based on the data above: "did he win more at the australian open or indian wells?". Execute SQL or Python code step-by-step and finally answer the question. Use SQL to process the query and use Python to reformat the data.

SQL: ```SELECT name, career_win_loss FROM DF WHERE name="Australian Open" or name="Indian Wells";```.

The database table DF is shown as follows:
[HEAD]: career_win_loss
---
[ROW] 1: Australian Open|22–18
[ROW] 2: Indian Wells|16-13

Answer the following question based on the data above: "did he win more at the australian open or indian wells?". Execute SQL or Python code step-by-step and finally answer the question. Use SQL to process the query and use Python to reformat the data.

Answer: ```Australian Open```.

The database table DF is shown as follows:
[HEAD]: by_race|white|black|aian*|asian|nhpi*
---
[ROW] 1: 2000 (total population)|75.43%|4.46%|19.06%|5.24%|0.88%
[ROW] 2: 2000 (Hispanic only)|3.42%|0.33%|0.45%|0.16%|0.06%
[ROW] 3: 2005 (total population)|74.71%|4.72%|18.77%|5.90%|0.88%
...
[ROW] 6: Growth 2000–05 (non-Hispanic only)|3.49%|11.30%|4.02%|18.96%|5.86%
[ROW] 7: Growth 2000–05 (Hispanic only)|33.56%|21.02%|14.52%|27.89%|-1.95%

Answer the following question based on the data above: "which hispanic population had the greatest growth from 2000 to 2005?". Execute SQL or Python code step-by-step and finally answer the question. Use SQL to process the query and use Python to reformat the data.

"""
    res = GptCompletion("gpt-3.5-turbo", prompt)
    print(res)