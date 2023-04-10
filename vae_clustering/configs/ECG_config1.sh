python3.7 ../train.py \
--lambda1 0.5 \
--lambda2 0.5 \
--device "cuda:1" \
--dataset "ECG" \
--train_set_path "../dataset_ECG/train_set.pt" \
--valid_set_path "../dataset_ECG/valid_set.pt" \
--recon_loss_type "mse" \
--loss3_type "triplet" \
--triplet_margin 2. \
--txt_result_prefix "../logs/txt/4.9/ECG_config1" \
--tb_result_dir "../logs/tb/4.9/ECG_config1" \
--model_prefix "../models/4.9/ECG_config1/VAE" \
--optimizer_prefix "../models/4.9/ECG_config1/VAE_op" \
--encoder_dim 32 \
--decoder_dim 32 \
--dim_ffn 64 \
--num_classes 5 \
--seq_len 140 \
--num_heads 8