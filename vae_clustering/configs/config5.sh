python3.7 ../train.py \
--lambda1 0.5 \
--lambda2 0.5 \
--device "cuda:1" \
--loss3_type "triplet" \
--triplet_margin 2. \
--txt_result_prefix "../logs/txt/3.29/config5" \
--tb_result_dir "../logs/tb/3.29/config5" \
--model_prefix "../models/3.29/config5/VAE" \
--optimizer_prefix "../models/3.29/config5/VAE_opt"