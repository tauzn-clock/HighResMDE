python inference_single_image.py  \
    --encoder large07 \
    --checkpoint_path ./models/1213_0154/nddepth_nyu/model-1000-best_d1_0.81294 \
    --image_path /scratchdata/nyu_depth_v2/official_splits/test/bathroom/rgb_00045.jpg \
    --max_depth 10.0 