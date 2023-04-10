python3.7 ../train.py \
--device "cuda:1" \
--loss3_type "category" \
--txt_result_prefix "../logs/txt/3.17/config1" \
--tb_result_dir "../logs/tb/3.17/config1" \
--model_prefix "../models/3.17/config1/VAE" \
--optimizer_prefix "../models/3.17/config1/VAE_opt" \
--batch_size 16 \
--validation_batch_size 16