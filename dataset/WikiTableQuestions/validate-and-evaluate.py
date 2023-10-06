#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate a WikiTableQuestions prediction file and run the official evaluator.

Usage:
    python validate-and-evaluate.py SPLIT PREDICTIONS \\
        [-d DATASET_DIR] [-o OUTPUT_FILE] [-c COMPARISON_FILE]

    SPLIT: Dataset split to evaluate on; can be one of the following:
        "test": Evaluate on pristine-unseen-tables
        "u-1", ..., "u-5": Evaluate on random-split-{1,2,3,4,5}-dev
    PREDICTIONS: Prediction file
        Each line must contain
            ex_id <tab> item1 <tab> item2 <tab> ...
        where ex_id is the example ID and item1, item2, ... is the predicted answer.
        If the model does not produce a prediction for an example, there must still
        be a line with just the ex_id and without the answer.
    DATASET_DIR: Path to the WikiTableQuestions dataset release.
    OUTPUT_FILE: Path to save the output JSON
    COMPARISON_FILE: Path to save the comparisons between gold and predicted answers.
"""

import sys, os, argparse, json, subprocess
from codecs import open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', default='WikiTableQuestions',
            help='Path to the dataset directory')
    parser.add_argument('-c', '--comparison-file', default='comparison.tsv',
            help='Print comparisons between gold and predicted answers to this file')
    parser.add_argument('-o', '--output-file', default='out.json',
            help='Dump output JSON to this file')
    parser.add_argument('split', choices=['u-1', 'u-2', 'u-3', 'u-4', 'u-5', 'test'],
            help='The dataset split used')
    parser.add_argument('predictions',
            help='Prediction file')
    args = parser.parse_args()

    output = {'split': args.split}

    # Check the dataset version
    with open(os.path.join(args.dataset_dir, 'README.md')) as fin:
        fin.readline()
        fin.readline()
        output['version'] = fin.readline().strip()
    print 'Dataset version: "{}"'.format(output['version'])

    # Get the example IDs for the split
    if args.split == 'test':
        print "using test split: pristine-unseen-tables.tsv"
        data_filename = 'pristine-unseen-tables.tsv'
    else:
        data_filename = 'random-split-{}-dev.tsv'.format(args.split[2])
    data_ex_ids = []
    with open(os.path.join(args.dataset_dir, 'data', data_filename)) as fin:
        print "Reading split file: ", data_filename
        fin.readline()      # Skip header
        for line in fin:
            data_ex_ids.append(line.split('\t')[0])
    print 'Read {} example IDs'.format(len(data_ex_ids))
    print 'Example ids: ', data_ex_ids[0:10]
    # Read the predictions
    prediction_ex_ids = set()
    with open(args.predictions, 'r', 'utf8') as fin:
        print "Using prediction file: ", args.predictions
        for line in fin:
            ex_id = line.strip().split('\t')[0].replace(' ', '')
            if ex_id in data_ex_ids:
                assert ex_id in data_ex_ids,\
                        'ERROR: Example {} is in the prediction file, ' \
                        'but not in the test data.'.format(ex_id)
                prediction_ex_ids.add(ex_id)
            else:
                print "Example {} is in the prediction file, but not in the test data".format(ex_id[0:100])
    
    for ex_id in data_ex_ids:
        assert ex_id in prediction_ex_ids,\
                'ERROR: Example {} is in the test data, '\
                'but not in the prediction file.'.format(ex_id)

    # Run the evaluator script
    with open(args.comparison_file, 'w') as fout:
        proc = subprocess.Popen(['python2',
            os.path.join(args.dataset_dir, 'evaluator.py'),
            '-t', os.path.join(args.dataset_dir, 'tagged', 'data'),
            args.predictions],
            stdout=fout,
            stderr=subprocess.PIPE)
        _, summary = proc.communicate()

    print summary
    for line in summary.split('\n'):
        tokens = line.strip().split(': ')
        if tokens[0] == 'Examples':
            output['examples'] = int(tokens[1])
        elif tokens[0] == 'Correct':
            output['correct'] = int(tokens[1])
        elif tokens[0] == 'Accuracy':
            output['accuracy'] = float(tokens[1])
    with open(args.output_file, 'w') as fout:
        json.dump(output, fout)


if __name__ == '__main__':
    main()

