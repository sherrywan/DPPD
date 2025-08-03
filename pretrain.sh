# finetune with initialization
python pretrain_motionbert.py \
--config configs/pretrain/PM_finetune.yaml \
-p /opt/data/private/3d_pose/Part_Motion/checkpoint/depth-2-dataratio-1/ \
-ms latest_epoch.bin
# -r /opt/data/private/3d_pose/Part_Motion/checkpoint/depth-2-dataratio-1/latest_epoch.bin