
experiment=experiment2

python run_test.py \
    --path_to_data_dir ../../../scratch/boesch/ata/Arena_Data/shuffle-3_split-test.npz \
    --dataset arena \
    --embedsum False \
    --fast_inference False \
    --batch_size 512 \
    --model gen_hiera \
    --input_size 900 1 54 \
    --stages 3 4 5 \
    --q_strides "5,1,1;5,1,1" \
    --mask_unit_attn True False False \
    --patch_kernel 1 1 54 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 128 192 256 \
    --distributed \
    --num_frames 900 \
    --num_workers 0 \
    --output_dir outputs/arena/${experiment}


cd hierAS-eval
    
nr_submissions=$(ls ../outputs/arena/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))
