
experiment=experiment4

python run_test.py \
    --path_to_data_dir /scratch/izar/boesch/data/Arena_Data/shuffle-3_split-test.npz \
    --dataset arena \
    --embedsum False \
    --fast_inference True \
    --batch_size 512 \
    --model gen_hiera \
    --input_size 1000 1 54 \
    --stages 2 3 4 \
    --q_strides "10,1,1;10,1,1" \
    --mask_unit_attn True False False \
    --patch_kernel 2 1 54 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 92 128 256 \
    --distributed \
    --num_frames 1000 \
    --num_workers 8 \
    --output_dir /scratch/izar/boesch/BehaveMAE/outputs/arena/${experiment}


cd hierAS-eval
    
nr_submissions=$(ls /scratch/izar/boesch/BehaveMAE/outputs/arena/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))
