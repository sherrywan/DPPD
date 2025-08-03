# finetune with initialization
python analysis.py \
--config configs/pretrain/PM_analysis.yaml \
--modeltype D3DP \
--chamfer True \
-p /opt/data/private/gait/PerNGR/checkpoint/diff_v0/ \
-ms best_epoch.bin