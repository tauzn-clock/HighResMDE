python eval.py  \
    --encoder large07 \
    --checkpoint_path /HighResMDE/nddepth_custom/models/1213_0308/nddepth_nyu/model-5000-best_d1_0.89619 \
    \
    --dataset nyu \
    --input_height 480 \
    --input_width 640 \
    --max_depth 10 \
    \
    --data_path_eval  /scratchdata/nyu_depth_v2/official_splits/test/ \
    --gt_path_eval /scratchdata/nyu_depth_v2/official_splits/test/ \
    --filenames_file_eval nyudepthv2_test_files_with_gt.txt \
    --min_depth_eval 1e-3 \
    --max_depth_eval 10 \
    --eigen_crop