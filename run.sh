export NGPUS=1
python3 tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" TEST.IMS_PER_BATCH images_per_gpu x $GPUS