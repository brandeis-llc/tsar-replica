import json
import sys
from pathlib import Path

import torch
from transition_amr_parser.parse import AMRParser


def parse_rams(parser, rams_jsonl):
    data = []
    out_txt_fname = rams_jsonl.with_suffix('.amr.txt')
    out_pkl_fname = rams_jsonl.with_suffix('.amr.pkl')
    with open(rams_jsonl) as in_f, open(out_txt_fname, 'w') as out_f:
        for line in in_f:
            doc = json.loads(line)
            did = doc['doc_key']
            for i, tokens in enumerate(doc['sentences']):
                print('\n\n', ' '.join(tokens))
                try:
                    penman, _ = parser.parse_sentence(tokens, jamr=True)
                # for sentences that parser can't handle and returns rootless penman
                # e.g., # ::tok ADVERTISEMENT
                except KeyError:
                    penman = f'# ::tok {" ".join(tokens)}\n()'
                out = f"snt_id::{did}::{i}\n{penman}\n\n"
                data.append(out)
                out_f.write(out)
                out_f.flush()
    torch.save(data, out_pkl_fname)


def parse_wikievents(parser, split):
    data = []
    with open('../data/wikievents/transfer-{}.jsonl'.format(split)) as f:
        for line in f:
            data.append(json.loads(line))

    all_sentences = []
    for d in data:
        sentences = d['sentences']
        all_sentences.extend(sentences)
        
    with open('amr-wikievent-{}.txt'.format(split), 'w') as f:
        amr_list = parser.parse_sentences(all_sentences)
        for res in amr_list:
            f.write(res+'\n\n')
    torch.save(amr_list, 'amr-wikievent-{}.pkl'.format(split))

if __name__ == '__main__':
    parser = AMRParser.from_pretrained('AMR3-structbart-L')
    rams_data_dir = Path(sys.argv[1])
    for split in ['train', 'dev', 'test']:
        parse_rams(parser, rams_data_dir / f'{split}.jsonlines')
