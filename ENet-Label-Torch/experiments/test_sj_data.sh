#!/usr/bin/env sh
exp=vgg_SCNN_DULR_w9
data=./data/sj_test_data
#rm gen/lane.t7
qlua testLane.lua \
	-model experiments/pretrained/ENet-label-new.t7 \
	-data ${data} \
	-val ${data}/test.txt \
	-save experiments/predicts/${exp}/sj_test_data \
	-dataset laneTest \
	-shareGradInput true \
	-nThreads 1 \
	-nGPU 2 \
	-batchSize 1 \
	-smooth true 
