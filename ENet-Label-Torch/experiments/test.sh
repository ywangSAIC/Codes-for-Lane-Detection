#!/usr/bin/env sh
exp=vgg_SCNN_DULR_w9
data=./data/CULane
#rm gen/lane.t7
qlua testLane.lua \
	-model experiments/pretrained/ENet-label-new.t7 \
	-data ${data} \
	-val ${data}/list/test.txt \
	-save experiments/predicts/${exp} \
	-dataset laneTest \
	-shareGradInput true \
	-nThreads 2 \
	-nGPU 2 \
	-batchSize 1 \
	-smooth true 
