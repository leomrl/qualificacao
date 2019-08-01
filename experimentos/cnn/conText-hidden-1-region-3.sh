#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

gpu=-1 # <= change this to, e.g., "gpu=0" to use a specific GPU.
mem=10

# pre-allocate 8GB device memory
gpumem=${gpu}:${mem}

prep_exe=~/conText-v4.00/bin/prepText
exe=~/conText-v4.00/bin/reNet

options="LowerCase UTF8"
#Generate vocabulary
echo Generating vocabulary from training data ...

max_num=10000
vocab_fn=data/pan_br_trn-${max_num}.vocab

#stopword_fn=data/stopwords

$prep_exe gen_vocab input_fn=data/pan_br_train.txt.tok vocab_fn=$vocab_fn max_vocab_size=$max_num \
$options WriteCount

#Generate region files (data/*.xsmatvar) and target files (data/*.y) for training and testing CNN.
echo Generating region files with region size 2 and 3 ...
for pch_sz in 1 2 3; do
for set in train test; do
	rnm=data/pan_br_${set}-patch${pch_sz}
	$prep_exe gen_regions \
	region_fn_stem=$rnm input_fn=data/pan_br_${set} vocab_fn=$vocab_fn \
	$options text_fn_ext=.txt.tok label_fn_ext=.cat \
	label_dic_fn=data/pan_br.dic \
	patch_size=$pch_sz patch_stride=1 padding=$((pch_sz-1))
	done
done
#Training and testing
log_fn=log_output/pan-seq.log
perf_fn=perf/pan-seq-perf.csv

echo
echo Training CNN and testing ...
echo This takes a while. See $log_fn and $perf_fn for progress and see param/seq-bow.param for the rest of the parameters.

nodes=2000 # number of neurons (weight vectors) in the convolution layer.

$exe $gpumem train extension=multi conn0=0-top conn1=1-top conn2=2-top \
data_dir=data trnname=pan_br_train- tstname=pan_br_test- \
reg_L2=0 top_reg_L2=1e-4 step_size=0.05 top_dropout=0.5 \
nodes=$nodes resnorm_width=$nodes \
LessVerbose test_interval=1 \
evaluation_fn=$perf_fn save_fn=output/pan_br_model \
loss=Square num_iterations=100 step_size_scheduler=Few \
step_size_decay=0.1 step_size_decay_at=80_90 mini_batch_size=100 \
0dataset_no=0 1dataset_no=1 2dataset_no=2 data_ext0=patch1 \
data_ext1=patch2 data_ext2=patch3 \
layers=3 pooling_type=Max num_pooling=1 activ_type=Rect \
random_seed=1 datatype=sparse_multi x_ext=.xsmatbcvar y_ext=.y \
momentum=0.9 init_weight=0.01 init_intercept=0 \
optim=Rmsp \
resnorm_type=Cross resnorm_alpha=1 resnorm_beta=0.5 > ${log_fn}

#optim=Rmsp \