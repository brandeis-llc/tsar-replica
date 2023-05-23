import json
import sys
from collections import defaultdict as ddict
from pathlib import Path

from tqdm import tqdm


def make_meta_rams(rams_data_dir):
    src_path = rams_data_dir / 'train.jsonlines'
    tgt_path = rams_data_dir / 'meta.json'
    
    eventtype2role = ddict(set)
    
    cut = 0
    with open(src_path) as in_f:
        for jl in in_f:
            if cut > 1000000:
                break
            cut += 1
            
            j = json.loads(jl)
            links = ddict(set)
            for link in j["gold_evt_links"]:
                evt_idx, _, link_type = link
                links[evt_idx[0]].add(link_type)
            for evt_idx, _, evt_types in j['evt_triggers']:
                for evt_type, _ in evt_types:
                    if evt_idx not in links:
                        pass
                        # print('xxx')
                    else:
                        for link_type in links[evt_idx]:
                            eventtype2role[evt_type].add(link_type)
    result = []
    for k, v in eventtype2role.items():
        r = [k, []]
        for vv in v:
            r[-1].append(vv)
        result.append(r)

    with open(tgt_path, 'w') as f:
        json.dump(result, f)


def make_meta_wikievents(src_path, tgt_path):
    data = []
    eventtype2role = dict()

    with open(src_path) as f:
        for line in f:
            data.append(json.loads(line))
    for d in tqdm(data):
        event_mentions = d['event_mentions']
        for event_mention in event_mentions:
            event_type = event_mention['event_type']
            if event_type not in eventtype2role:
                eventtype2role[event_type] = set()
            arguments = event_mention['arguments']
            for argument in arguments:
                role = argument['role']
                eventtype2role[event_type].add(role)

    result = []
    for k, v in eventtype2role.items():
        r = [k, []]
        for vv in v:
            r[-1].append(vv)
        result.append(r)

    with open(tgt_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    # use TRAINING SET to make meta 
    rams_data_dir = sys.argv[1]
    make_meta_rams(Path(rams_data_dir))
    # make_meta_wikievents('wikievents/train.jsonl', 'meta.json')
