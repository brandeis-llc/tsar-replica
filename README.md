# TSAR replica

README to keep notes about attempts to run TSAR code without too much of changing the code base.

## Dependencies

Everything is done with python 3.10

### For AMR parsing 
The original paper used 2020 version of IBM parser ([stack-transformat](https://github.com/IBM/transition-amr-parser/tree/stack-transformer)), however running that version on _modern_ CUDA versions wasn't so easy. So I just decided to run the latest version of the IBM parser (in the `main` branch).

- `transition-neural-parser`==0.5.4 (comes with `torch`==1.13)
- `torch-scatter`

### For preprocessing and training
- `pytorch`==1.11.0 
  - torch 1.9 prebuilt packages do not support recent cuda versions
  - building torch from the source code is too much hassle 
  - ended up running 1.11 with CUDA 11.3
- `transformers`==4.8.1
- `datasets`==2.13.0
- `dgl-cu111`==1.1.0+cu113
- `tqdm`==4.65.0
- `spacy`==3.2.4

### Preprocessing

#### Data acquisition

> You can first download the datasets and some scripts [here](https://drive.google.com/file/d/1euuD7ST94b5smaUFo6ROLW_ZasHwDpib/view?usp=sharing).
> You only need to unzip the data.zip.

Data comes in two directories; `wikievents` and `rams`. I think `wikievents` is not relevant. 

RAMS data included in the zip file is actually identical to [RAMS 1.0c from the official website](https://nlp.jhu.edu/rams/), except for the `meta.json` and `result.json` files. 
(**TODO** RAMS 1.0c contains scorer scripts that fix some bug in the previous version. We might need to check if TSAR is using those scorer scripts.) 

* `meta.json`: The wikievents data in the zip file also has `meta.json` file, and there's `make_meta.py` script that takes wikievents dataset and generate that `meta.json` file. For RAMS, there's `meta.json` but no `make_meta` script, so I re-created `make_meta.py` for RAMS based on the contents of the `meta.json` file. 
* `result.json`: Not sure what it is. Searching in the original codebase (github repo and local clone) didn't give any clue.

#### AMR parsing

As mentioned above, the original parser wasn't easy to install, so I used the latest IBM parser to generate penman + jamr formatted `.amr.txt` files. 

#### DGL conversion 

> We also directly provide the data (used in the original paper) [here](https://drive.google.com/drive/folders/1GBmvZJJP6f0jUmFaAuvk_q7Nbw_lElH0?usp=sharing).
> In this way, you can just skip the AMR and DGL graph preprocessing steps.
> If you want to run this model with GL events or remake any of the DGL graphs based on different edge clusters, you will need to run the whole preprocessing pipeline starting from the AMR .txt or .pkl files.
 
The data in the link is supposedly saved with torch 1.9 + dgl 0.6, which doesn't load with torch 1.13 + dgl 1.1.0. So I have to [slight edit](https://github.com/keighrim/tsar-replica/commit/59619d3aab26ddf3516e96cd1af9d5913196536b) `amr2dgl` script to load newly generate `.amr.txt` files. 

### Training and Evaluation

> The training scripts are provided.
> 
> ```bash
> bash run_rams_base.sh <data-directory>
> bash run_rams_large.sh <data-directory>
> bash run_wikievents_base.sh <data-directory>
> bash run_wikievents_large.sh <data-directory>
```

Running `run_rams_base.sh` originally hit a out-of-index error at where the original authors acknowledge a possible problem. 

https://github.com/RunxinXu/TSAR/blob/9806edfb5a7f90b9ae85ff06f435c20e4222be59/code/run.py#L443-L444
