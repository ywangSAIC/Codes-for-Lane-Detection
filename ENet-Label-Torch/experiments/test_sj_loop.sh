#!/usr/bin/env sh
exp=vgg_SCNN_DULR_w9
data=./data/sj_big_loop
#rm gen/lane.t7
qlua testLane.lua \
	-model experiments/pretrained/ENet-label-new.t7 \
	-data ${data} \
	-val ${data}/test.txt \
	-save experiments/predicts/${exp}/sj_big_loop\
	-dataset laneTest \
	-shareGradInput true \
	-nThreads 2 \
	-nGPU 2 \
	-batchSize 10 \
	-smooth true 


ffmpeg -r 30 -s 976x208 -pattern_type glob -i "experiments/predicts/${exp}/sj_big_loop/*_ret.png" -vcodec libx264 -pix_fmt yuv420p experiments/predicts/${exp}/sj_big_loop/result.MP4
