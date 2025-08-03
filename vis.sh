python vis.py \
--config configs/diffusion/PM_diff_test.yaml \
-Mc checkpoint/motionbert/best_epoch.bin \
-Dc checkpoint/diff \
--evaluate best_epoch.bin \
--feature_loss True \
--first_frame_loss True \
-num_proposals 5 \
-sampling_timesteps 5 \
-b 1