#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

gpu=-1 # <= change this to, e.g., "gpu=0" to use a specific GPU.
mem=8

# pre-allocate 8GB device memory
gpumem=${gpu}:${mem}

exe=~/conText-v4.00/bin/reNet

$exe -1 predict model_fn=output/pan_br_model.epo100.ReNet \
prediction_fn=output/pan_prediction.txt WriteText extension=multi \
datatype=sparse_multi tstname=pan_br_test- data_ext0=patch1 \
data_ext1=patch2 data_ext2=patch3 data_dir=data x_ext=.xsmatbcvar > output-test
