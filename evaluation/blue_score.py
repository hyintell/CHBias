import nltk
from nltk.translate.bleu_score import SmoothingFunction
import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument('--hyp_file', help='Path containing hypothesis file')
parser.add_argument('--ref_file', help='Path containing reference file')

args = parser.parse_args()

hyp_file_path = args.hyp_file
ref_file_path = args.ref_file

# hyp_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/response_hyp.txt'
# ref_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/test.refs.txt'

#with open(hyp_file_path, encoding="utf-8") as f:
#    hyp_lines = [line for line in f.read()]

with open(hyp_file_path) as f:
    hyp_lines = json.load(f)

print('First hyp line {}'.format(hyp_lines[0]))

with open(ref_file_path, encoding="utf-8") as f:
    ref_lines = [line for line in f.read().splitlines()]

print('First Reference line {}'.format(ref_lines[0]))

hyp_sentences = []
ref_sentences = []
dict_ref = {}

for ref in ref_lines:
    #print('ref is {}'.format(ref))
    key = ref.split('\t')[0]
    #print('key {}'.format(key))
    dict_ref[key] = ref.split('\t')[-1]
    # response = ref['response'].replace(ref['input'], '')
    # key = ref['key']
    # dict_ref[key] = response

for line in hyp_lines:
    #print('line {}'.format(line))
    # key = line.split(': ')[0]
    # key = key.strip('{"')
    # hyp_sent = line.split(': ')[1]
    hyp_sent = line['response'].replace(line['input'], '')
    key = line['key']
    ref_sent = dict_ref[key]
    #print('hyp_sent {}'.format(hyp_sent))
    #print('ref_sent {}'.format(ref_sent))
    ref_s = re.sub(r"[^a-zA-Z0-9]+", ' ', ref_sent).split()
    hyp_s = re.sub(r"[^a-zA-Z0-9]+", ' ', hyp_sent).split()
    ref_sentences.append([ref_s])
    hyp_sentences.append(hyp_s)

print('number of hyp {}'.format(len(hyp_sentences)))
print('number of ref {}'.format(len(ref_sentences)))
print('hyp sentence {}'.format(hyp_sentences[:3]))
print('ref sentences {}'.format(ref_sentences[:3]))
#smooth = SmoothingFunction().method4
chencherry = SmoothingFunction()

bleu_cor = nltk.translate.bleu_score.corpus_bleu(ref_sentences, hyp_sentences, weights=(0.5, 0.5), smoothing_function=chencherry.method1)
print('bleu for corpus: {}'.format(bleu_cor))

print('Calculating average of sentence BLEU.. ')

total_score = 0
for ref_l, hyp_l in zip(ref_sentences, hyp_sentences):
    #print(ref_l, hyp_l)
    bleu_sent = nltk.translate.bleu_score.sentence_bleu(ref_l, hyp_l, weights=(0.75, 0.25), smoothing_function=chencherry.method1)
    #print('Bleu score {}'.format(bleu_sent))
    total_score = total_score + bleu_sent

print('Average BLEU score {}'.format(total_score/len(hyp_sentences))) 
