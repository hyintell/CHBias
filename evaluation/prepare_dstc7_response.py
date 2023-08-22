import nltk
from nltk.translate.bleu_score import SmoothingFunction
import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument('--hyp_file', help='Path containing hypothesis file')
parser.add_argument('--ref_file', help='Path containing reference file')
parser.add_argument('--dest_file', help='dest file')

args = parser.parse_args()

hyp_file_path = args.hyp_file
ref_file_path = args.ref_file
dest_path = args.dest_file
# hyp_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/response_hyp.txt'
# ref_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/test.refs.txt'


with open(hyp_file_path) as f:
    hyp_lines = json.load(f)

print('First hyp line {}'.format(hyp_lines[0]))

with open(ref_file_path, encoding="utf-8") as f:
    ref_lines = [line for line in f.read().splitlines()]

print('First Reference line {}'.format(ref_lines[0]))

f = open(dest_path, 'w')
data = ''

for ref in ref_lines:
    ref_key = ref.split('\t')[0]

    for line in hyp_lines:
        #print('line {}'.format(line))
        # key = line.split(': ')[0]
        # key = key.strip('{"')
        # hyp_sent = line.split(': ')[1]
        if ref_key == line['key']:
            hyp_sent = line['response'].replace(line['input'], '')
            break
    new_ref = ref.replace('__UNDISCLOSED__' , hyp_sent)
    data += new_ref + '\n'

f.write(data)

