# inc-few-shot-attractor-public

This repository contains code for the following paper:
**Incremental Few-Shot Learning with Attention Attractor Networks**. Mengye Ren, Renjie Liao, Ethan Fetaya, Richard S. Zemel. NeurIPS 2019. [[arxiv](https://arxiv.org/abs/1810.07218)]

## Dependencies
* cv2
* numpy
* pandas
* python 2.7 / 3.5+
* tensorflow 1.11
* tqdm

Our code is tested on Ubuntu 14.04 and 16.04.

## Setup
First, designate a folder to be your data root:
```
export DATA_ROOT={DATA_ROOT}
```

Then, set up the datasets following the instructions in the subsections.

### miniImageNet
[[Google Drive](https://drive.google.com/open?id=13DV4S4hyc1zLomr8Ej3YwQMyAlkNui-8)]  (5.5 GB)
```
# Download and place "mini-imagenet.tar.gz" in "$DATA_ROOT/mini-imagenet".
mkdir -p $DATA_ROOT/mini-imagenet
cd $DATA_ROOT/mini-imagenet
mv ~/Downloads/mini-imagenet.tar .
tar -xvf mini-imagenet.tar
rm -f mini-imagenet.tar
```

### tieredImageNet
[[Google Drive](https://drive.google.com/open?id=1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u)]  (12.9 GB)
```
# Download and place "tiered-imagenet.tar" in "$DATA_ROOT/tiered-imagenet".
mkdir -p $DATA_ROOT/tiered-imagenet
cd $DATA_ROOT/tiered-imagenet
mv ~/Downloads/tiered-imagenet.tar .
tar -xvf tiered-imagenet.tar
rm -f tiered-imagenet.tar
```
Note: Please make sure that the following hardware requirements are met before
running tieredImageNet experiments.
* Disk: **30 GB**
* RAM: **32 GB**

### Config files
Run make to make protobuf files.
```
git clone https://github.com/renmengye/inc-few-shot-attractor.git
cd inc-few-shot-attractor
make
```

## Core Experiments

### Pretraining
```
./run.sh {GPUID} python run_exp.py --config {CONFIG_FILE}     \
                  --dataset {DATASET}                         \
                  --data_folder {DATASET_FOLDER}              \
                  --results {SAVE_FOLDER}                     \
                  --tag {EXPERIMENT_NAME}
```
* Possible `DATASET` options are `mini-imagenet`, `tiered-imagenet`.
* Possible `CONFIG` options are any prototxt file in the `./configs/pretrain`
  folder.

### Meta-learning
```
./run.sh {GPUID} python run_exp.py --config {CONFIG_FILE}     \
                  --dataset {DATASET}                         \
                  --data_folder {DATASET_FOLDER}              \
                  --pretrain {PRETRAIN_CKPT_FOLDER}           \
                  --nshot {NUMBER_OF_SHOTS}                   \
                  --nclasses_b {NUMBER_OF_FEWSHOT_WAYS}       \
                  --results {SAVE_FOLDER}                     \
                  --tag {EXPERIMENT_NAME}                     \
                  [--eval]                                    \
                  [--retest]
```
* Possible `DATASET` options are `mini-imagenet`, `tiered-imagenet`.
* Possible `CONFIG` options are any prototxt file in the `./configs/attractors`
  folder, e.g. `\*-{mlp|lr}-attn-s{1|5}.prototxt` means 1/5-shot model using
  MLP or LR as fast weights model.
* You need to pass in `PRETRAIN_CKPT_FOLDER` option with the pretrained model.
* Add `--retest` flag for restoring a fully trained model and re-run eval.

### Baselines
* Baseline configs are in `./configs/lwof` and `./configs/imprint`.
* For ProtoNet baseline please run `run_proto_exp.py` with the same flags from
  the previous section.
* Configs for ablation studies can be found in `./configs/ablation`.

## Citation
If you use our code, please consider cite the following:
* Mengye Ren, Renjie Liao, Ethan Fetaya and Richard S. Zemel.
Incremental Few-Shot Learning with Attention Attractor Networks.
In *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

```
@inproceedings{ren19incfewshot,
  author   = {Mengye Ren and
              Renjie Liao and
              Ethan Fetaya and
              Richard S. Zemel},
  title    = {Incremental Few-Shot Learning with Attention Attractor Networks,
  booktitle= {Advances in Neural Information Processing Systems (NeurIPS)},
  year     = {2019},
}
```
