python3.7 ../train.py \
--device "cuda:2" \
--loss3_type "triplet" \
--txt_result_prefix "../logs/txt/3.17/config2" \
--tb_result_dir "../logs/tb/3.17/config2" \
--model_prefix "../models/3.17/config2/VAE" \
--optimizer_prefix "../models/3.17/config2/VAE_opt" \
--batch_size 16 \
--validation_batch_size 16