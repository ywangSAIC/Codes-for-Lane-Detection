python3 -u test_erfnet.py CULane ERFNet train test_img \
                          --dataset_path /home/ywang/dataset/sj_big_loop/list \
                          --lr 0.01 \
                          --gpus 0 1 \
                          --npb \
                          --resume trained/ERFNet_trained.tar \
                          --img_height 208 \
                          --img_width 976 \
                          -j 16 \
                          -b 10
