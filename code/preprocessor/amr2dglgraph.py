import torch
import dgl
import json
from tqdm import tqdm

PB_VN_MAPPINGS = json.load(open("../code/preprocessor/pb-vn2.json"))
PREDICATE = ""


# Current Configuration: GL only, AMR all as default
def get_amr_edge_idx(edge_type_str):

    # remove ?
    edge_type_str = edge_type_str.replace("?", "")
    if edge_type_str.startswith("ARG"):
        return 9


    # if the edge is in event-structure (came from VN)
    if (edge_type_str.upper() == edge_type_str) and (edge_type_str != "E") and (edge_type_str != ""):
        # if the predicate is not in verb net, return the default edge
        if (pred := PREDICATE.replace("-", ".")) not in PB_VN_MAPPINGS:
            return 9

        pb_vns = list(PB_VN_MAPPINGS[pred].values())

        # some have no arguments
        pb_vns = [p for p in pb_vns if p]

        if len(pb_vns) == 0:
            return 9
        else:
            val_lists = [list(p.values()) for p in pb_vns]

            to_del = []
            for idx, vals in enumerate(val_lists):
                if edge_type_str.lower() not in vals:
                    to_del.append(idx)

            to_del = to_del[::-1]
            for idx in to_del:
                del pb_vns[idx]

            # did not have a corresponding argument, return default edge
            if len(pb_vns) == 0:
                return 9
            # case where the current edge node disambiguates the argument structure
            else:
                pb_vn = pb_vns[0]

        vn_pb = {v: k for k, v in pb_vn.items()}

        edge_type_str = vn_pb[edge_type_str.lower()]

    # GL edges
    if edge_type_str.startswith("subevent"):
        return 0
    elif edge_type_str.startswith("event-structure"):
        return 1
    # all other events
    elif edge_type_str.startswith("e") and edge_type_str[-1].isdigit():
        return 2
    # stative events that do not change
    elif edge_type_str == "E":
        return 3
    elif edge_type_str == 'ARG0':
        return 4
    elif edge_type_str == 'ARG1':
        return 5
    elif edge_type_str == 'ARG2':
        return 6
    elif edge_type_str == 'ARG3':
        return 7
    elif edge_type_str == 'ARG4':
        return 8
    else:
        return 9


def processing_amr(data, amr_list):
    '''
    把每个句子变成一个dglgraph
    编号0的是root节点
    ndata['span']放的是对应的word-level的span信息
    '''


    graphs_list = []
    cur_idx = 0
    initial_graph = {}
    for i in range(10):
        initial_graph[('node', str(i), 'node')] = ([], [])

    all_edge_type = {}
    for sentences in tqdm(data):
        amrs = amr_list[cur_idx:cur_idx + len(sentences)]
        cur_idx += len(sentences)
        graphs = []
        for sent, amr in zip(sentences, amrs):
            graph = dgl.heterograph(initial_graph)
            amrnodeid2span = {}
            graphnodeid2amrnodeid = {}
            amrnodeid2graphnodeid = {}
            root_amrnodeid = -1

            amr_split_list = amr.split('\n')
            for line in amr_split_list:
                if line.startswith('# ::node'):
                    node_split = line.split('\t')
                    amrnodeid = node_split[1]
                    if len(node_split) != 4:
                        # check if the alignment text spans exist
                        continue
                    else:
                        align_span = node_split[3].split('-')
                        if not align_span[0].isdigit() or not align_span[1].isdigit():
                            continue
                        start, end = int(align_span[0]), int(align_span[1]) - 1
                        amrnodeid2span[amrnodeid] = (start, end)
                elif line.startswith('# ::root'):
                    line = line.split('\t')
                    root_amrnodeid = line[1]

            if root_amrnodeid == -1:
                print('=======>  No AMR graph!!!!!')
                graph.add_nodes(num=1)
                graph.ndata['span'] = torch.zeros(1, 2, dtype=torch.long)
                graphs.append(graph)
                continue

            # root must be the first node
            # one exception is that root is not in amrnodeid2span
            if root_amrnodeid in amrnodeid2span:
                graphnodeid2amrnodeid[0] = root_amrnodeid
                amrnodeid2graphnodeid[root_amrnodeid] = 0
            else:
                root_amrnodeid = -1
            cur_id_num = len(graphnodeid2amrnodeid)
            for amrnodeid in amrnodeid2span:
                if amrnodeid != root_amrnodeid:
                    graphnodeid2amrnodeid[cur_id_num] = amrnodeid
                    amrnodeid2graphnodeid[amrnodeid] = cur_id_num
                    cur_id_num += 1

            # create nodes
            graph.add_nodes(num=len(amrnodeid2span))
            # create edges
            for line in amr_split_list:
                if line.startswith('# ::edge'):
                    edge_split = line.split('\t')
                    amr_edge_type = edge_split[2]
                    edge_start = edge_split[4]
                    edge_end = edge_split[5]
                    # check if the start and end nodes exist
                    if (edge_start in amrnodeid2span) and (edge_end in amrnodeid2span): 
                        # check if the edge type is "ARGx-of", if so, reverse the direction of the edge
                        if amr_edge_type.startswith("ARG") and amr_edge_type.endswith("-of"):
                            edge_start, edge_end = edge_end, edge_start
                            amr_edge_type = amr_edge_type[0:4]
                        # deal with this edge here
                        if amr_edge_type == "event-structure":
                            global PREDICATE
                            PREDICATE = edge_split[1]
                        edge_type = str(get_amr_edge_idx(amr_edge_type))
                        if amr_edge_type not in all_edge_type:
                            all_edge_type[amr_edge_type] = 0
                        all_edge_type[amr_edge_type] += 1

                        # forward
                        graph.add_edges(u=amrnodeid2graphnodeid[edge_start], v=amrnodeid2graphnodeid[edge_end], etype=edge_type)
                        # also backward
                        graph.add_edges(u=amrnodeid2graphnodeid[edge_end], v=amrnodeid2graphnodeid[edge_start], etype=edge_type)
            
            # add span features to node: word-level span
            features = torch.zeros(len(graphnodeid2amrnodeid), 2, dtype=torch.long)
            for i in range(len(graphnodeid2amrnodeid)):
                features[i][0] = amrnodeid2span[graphnodeid2amrnodeid[i]][0]
                features[i][1] = amrnodeid2span[graphnodeid2amrnodeid[i]][1]
            graph.ndata['span'] = features
            graphs.append(graph)

        graphs_list.append(graphs)

    print(all_edge_type)
    return graphs_list


def read_amr_txt(amr_txt_fname):
    amr = []
    with open(amr_txt_fname) as f:
        cur_amr = ""
        for line in f:
            if line.startswith('snt_id'):
                if cur_amr:
                    amr.append(cur_amr.strip())
                    cur_amr = ""
            if line:
                cur_amr += line
        if cur_amr:
            amr.append(cur_amr.strip())
    return amr


def amr2dglgraph(data_path, amr_path, graph_path):
    data = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            sentences = d['sentences']
            data.append(sentences)
    amr = torch.load(amr_path)
    graphs_list = processing_amr(data, amr)
    torch.save(graphs_list, graph_path)


if __name__ == "__main__":
    amr2dglgraph("train.jsonlines", "amr-rams-train.pkl", "dglgraph-rams-train.pkl")
    amr2dglgraph("dev.jsonlines", "amr-rams-dev.pkl", "dglgraph-rams-dev.pkl")
    amr2dglgraph("test.jsonlines", "amr-rams-test.pkl", "dglgraph-rams-test.pkl")
