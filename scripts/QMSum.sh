CUDA_VISIBLE_DEVICES=1 python ./src/run.py \
        -data_path=./qmsum_data/512 \
        -val_save_file=./data/qmsum/val_out/valid_out.txt \
        -test_save_file=./data/qmsum/test_results/bart.results \
        -model=bart-wikisum \
        -checkpoint=None \
        -log_name=logs/12_512_KA_in_both \
        -gpus='-1' \
        -batch_size=1 \
        -learning_rate=5e-6 \
        -num_epochs=10 \
        -warmup=0 \
        -grad_accum=1 \
        -random_seed=4326 \
        -do_train \
        -limit_val_batches=1 \
        -val_check_interval=1 \
        -number_of_segment=2 \
        -max_input_len=535 \
        -max_output_len=256 \
        -min_output_len=0 \
        -n_beams=5 \
        -length_penalty=1 \

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf


# -knowledge_aware=concat \

# -log_name=logs/8_512_dual_rank_kn \

# -data_path=./data/qmsum/processed_data_turn_split \
# -log_name=logs/8_512_second_stage_final \