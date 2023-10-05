CUDA_VISIBLE_DEVICES=1 python ./src/run.py \
        -data_path=./data/qmsum/512 \
        -val_save_file=./data/qmsum/val_out/valid_out.txt \
        -test_save_file=./data/qmsum/test_results/12_512_no_KA_in_generation.results \
        -model=bart-wikisum \
        -checkpoint=logs/12_512_no_KA_in_generation/version_13/checkpoints/epoch\=4-step\=6285.ckpt/ \
        -log_name=logs/test \
        -gpus='-1' \
        -batch_size=1 \
        -learning_rate=5e-6 \
        -num_epochs=20 \
        -warmup=0 \
        -grad_accum=1 \
        -random_seed=0 \
        -do_test \
        -limit_val_batches=1 \
        -val_check_interval=1 \
        -number_of_segment=8 \
        -max_input_len=535 \
        -max_output_len=256 \
        -min_output_len=0 \
        -n_beams=5 \
        -length_penalty=1 \
        
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# -knowledge_aware=concat \
# 3407
# -data_path=./data/qmsum/processed_data_turn_split \