#!/bin/bash

python generation/train_dreambooth.py \
    --pretrained_model_name_or_path='stabilityai/stable-diffusion-3-medium' \
    --instance_data_dir='./data/pods/train' \
    --class_data_dir='./synthetic_data/pods/pods_class_data_dir' \
    --output_dir='./finetuned_models' \
    --prior_loss_weight='1.0' \
    --instance_prompt='a photo of sks {sc}' \
    --class_prompt='A photo of {sc}' \
    --resolution='512' \
    --train_batch_size='1' \
    --gradient_accumulation_steps='1' \
    --learning_rate='2e-6' \
    --lr_scheduler='constant' \
    --lr_warmup_steps='0' \
    --num_class_images='200' \
    --validation_prompt='A photo of a sks {sc} on the beach' \
    --max_train_steps='400' \
    --train_text_encoder \
    --with_prior_preservation \
    --use_8bit_adam \
    --gradient_checkpointing \
    --class_name='mugs_blue_greek' \
    --class_category='mug' 
