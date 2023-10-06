import re
import pandas as pd
import os
import json
import numpy as np
import itertools
import random

def tokenizeDF(df, utterance=None, refine_utt_col_names=True):
    token_dict = {}
    token_id = 0
    cols = [c.lower() for c in df.columns]
    
    for rid in range(df.shape[0]):
        for dtype, c in zip(df.dtypes, df.columns):
            string = df.loc[rid][c] 
            if  isinstance(string, float) \
                or isinstance(string, int) \
                or isinstance(string, np.int64) \
                or bool(re.search(r'\d', string)):
                continue
            new_tokens = []
            
            # entire string is a single token
            # if string.lower() in token_dict:
            #     new_tokens.append(token_dict[string.lower()])
            # else:
            #     token_dict[string.lower()] = f'TOKEN{token_id}'
            #     new_tokens.append(token_dict[string.lower()])
            #     token_id += 1
            
            # every word is a token
            for token in string.replace('\n', ' ').replace('_', ' ').split(' '):
                if token.lower() in cols:
                    new_tokens.append(token)
                elif token.lower() in token_dict:
                    new_tokens.append(token_dict[token.lower()])
                else:
                    token_dict[token.lower()] = f'TOKEN{token_id}'
                    new_tokens.append(token_dict[token.lower()])
                    token_id += 1
                
            df.at[rid, c] = ' '.join(new_tokens)
    if utterance is not None:
        new_utterance = []
        for u in utterance.strip('?').split(' '):
            if u.lower() in cols and refine_utt_col_names:
                new_utterance.append(df.columns[cols.index(u.lower())])
            elif u.lower() in token_dict:
                new_utterance.append(token_dict[u.lower()] + ' ')
            else:
                new_utterance.append(u)
        utterance = ' '.join(new_utterance) + '?'
    
    return df, utterance, token_dict

def tokenizeDFWithColNames(df, utterance=None, refine_utt_col_names=True):
    token_dict = {}
    cols = [c.lower() for c in df.columns]
    token_ids = {k: 0 for k in cols}
    
    for rid in range(df.shape[0]):
        for dtype, c in zip(df.dtypes, df.columns):
            string = df.loc[rid][c] 
            if  isinstance(string, float) \
                or isinstance(string, int) \
                or isinstance(string, np.int64) \
                or bool(re.search(r'\d', string)):
                continue
            new_tokens = []
            
            # entire string is a single token
            # if string.lower() in token_dict:
            #     new_tokens.append(token_dict[string.lower()])
            # else:
            #     token_dict[string.lower()] = f'TOKEN{token_id}'
            #     new_tokens.append(token_dict[string.lower()])
            #     token_id += 1
            
            # every word is a token
            for token in string.replace('\n', ' ').replace('_', ' ').split(' '):
                if token.lower() in cols:
                    new_tokens.append(token)
                elif token.lower() in token_dict:
                    new_tokens.append(token_dict[token.lower()])
                else:
                    token_dict[token.lower()] = f'{c.upper().replace(" ", "_")}{token_ids[c.lower()]}'
                    new_tokens.append(token_dict[token.lower()])
                    token_ids[c.lower()] += 1
                
            df.at[rid, c] = ' '.join(new_tokens)
    if utterance is not None:
        new_utterance = []
        for u in utterance.strip('?').split(' '):
            if u.lower() in cols and refine_utt_col_names:
                new_utterance.append(df.columns[cols.index(u.lower())])
            elif u.lower() in token_dict:
                new_utterance.append(token_dict[u.lower()] + ' ')
            else:
                new_utterance.append(u)
        utterance = ' '.join(new_utterance) + '?'
    
    return df, utterance, token_dict

def parseTokenizedStr(s, token_dict, value2token=False):
    if not value2token:
        rev_dict = {v:k for k,v in token_dict.items()}
    else:
        rev_dict = {k.upper():v for k,v in token_dict.items()}
    
    new_str = []
    for t in s.split(' '):
        if t.upper() in rev_dict:
            new_str.append(rev_dict[t.upper()])
        else:
            new_str.append(t)
    return ' '.join(new_str)