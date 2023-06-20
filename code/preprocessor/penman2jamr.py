from code.preprocessor.amr_io import AMR
import torch


def generate_node_line(var: str, node_dict, align_dict, *, is_root: bool):
    if is_root:
        head = "# ::root"
        return "\t".join([head, var, node_dict[var]])
    else:
        head = "# ::node"
        # don't add alignment information to placeholders
        if align_dict.get(var) and node_dict[var] != "ROLE":
            # assume each node is aligned with a single token
            return "\t".join([head, var, node_dict[var], f"{align_dict[var][0]}-{align_dict[var][0] + 1}"])
        else:
            # some nodes from the subevents don't have alignments, can we leave it blank?
            return "\t".join([head, var, node_dict[var]])


def generate_edge_line(edge_tuple, node_dict):
    head = "# ::edge"
    node1 = node_dict[edge_tuple[0]]
    node2 = node_dict[edge_tuple[2]]
    edge = edge_tuple[1][1:]  # remove ":"
    return "\t".join([head, node1, edge, node2, edge_tuple[0], edge_tuple[2]])


def add_subevents_alignments(edges, alignments):
    for s, rel, e in edges:
        if rel == ":event-structure" and not alignments.get(e):
            alignments[e] = alignments[s]
    for s, rel, e in edges:
        if not alignments.get(e):
            alignments[e] = alignments[s]
    return alignments


def penman2jamr(penman_txt: str):
    jamr_lines = []
    graph = AMR.from_penman(penman_txt)

    alignments = add_subevents_alignments(graph.edges, graph.alignments)

    jamr_lines.append(generate_node_line(graph.root, graph.nodes, alignments, is_root=True))
    for node_var in graph.nodes:
        jamr_lines.append(generate_node_line(node_var, graph.nodes, alignments, is_root=False))
    for edge_tuple in graph.edges:
        jamr_lines.append(generate_edge_line(edge_tuple, graph.nodes))
    return jamr_lines


if __name__ == '__main__':
    penman_txt = """
    (i / illustrate-01~3
       :ARG0 (p3 / point-04~2
                 :ARG1-of (s2 / specific-02~1)
                 :quant "3"~0)
       :ARG1 (t / thing~4
                :ARG0-of (c / cause-01~4
                            :ARG1 (s / see-01~6
                                     :ARG0 (p / person~5
                                              :mod (c2 / country~5
                                                       :name (n / name~5
                                                                :op1 "America"~5)))
                                     :ARG1 (p2 / person~7
                                               :name (n2 / name~7
                                                         :op1 "Trump"~7))
                                     :ARG2 (p4 / problem~10)
                                     :event-structure (se1 / subevents
                                                           :e1 (p5 / perceive
                                                                   :EXPERIENCER s
                                                                   :STIMULUS p2)))))
       :event-structure (se / subevents
                            :e1 (b / be
                                   :polarity -
                                   :THEME t)
                            :e2 (c1 / create_image
                                    :AGENT i
                                    :THEME t)
                            :e3 (a / and
                                   :op1 (b1 / be
                                            :THEME t)
                                   :op2 (p1 / part_of
                                            :THEME t
                                            :DESTINATION (R / ROLE))
                                   :op3 (c3 / cause))))"""
    for l in penman2jamr(penman_txt):
        print(l)

    print("hi")

    for split in ["test", "dev", "train"]:
        with open("glamrs/{}.glamrs".format(split)) as f:
            content = f.read()

        # list of GLAMR in graph format
        result = content.split("\n\n\n")
        fails = 0
        saves = 0

        with open("full_{}.glamrs".format(split), "wt") as f:

            # get amrs for fallback in case GLAMR fails
            with open("amrs/{}.amr".format(split)) as fallback:
                fallback = fallback.read()

            amr_list = fallback.split("\n\n\n")
            amr_dict = {}

            for sentence in amr_list:
                tag = sentence.split("\n")[0]
                s = "\n".join(sentence.split("\n")[1:])
                amr_dict.update({tag: s})

            all_info = []
            for sentence in result:
                tag = sentence.split("\n")[0]
                s = "\n".join(sentence.split("\n")[1:])

                # no tag, no way to recover, just skip, likely the end of the document
                if tag == "":
                    continue

                # sentence ID but no graph, recover from AMR
                elif s == "":
                    jamr_lines = amr_dict[tag]
                    jamr_and_graph = tag + "\n" + jamr_lines

                # both are present
                else:
                    # sometimes this fails, if it does, recover from AMR
                    try:
                        jamr_lines = penman2jamr(s)
                    except Exception:
                        print("Failed: " + tag)
                        fails += 1
                        if tag != "":
                            try:
                                jamr_lines = amr_dict[tag].split("\n")
                            except Exception:
                                print("FUCK!!!")
                                break
                            else:
                                print("AMR Saved!")
                                saves += 1
                    jamr_and_graph = tag + "\n" + "\n".join(jamr_lines) + "\n" + "\n".join(sentence.split("\n")[1:])

                f.write(jamr_and_graph+"\n\n\n")
                all_info.append(jamr_and_graph)
        print("{} has {} fails and {} saves".format(split, fails, saves))
        fails = 0
        saves = 0

        torch.save(all_info, "amr-rams-{}.pkl".format(split))

    # jamr_lines = penman2jamr(penman_txt)
    # for line in jamr_lines:
    #     print(line)