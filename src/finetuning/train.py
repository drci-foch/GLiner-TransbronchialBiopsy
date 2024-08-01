import os
import argparse
import random
import json

from transformers import AutoTokenizer, Trainer, TrainingArguments
import torch

from gliner import GLiNERConfig, GLiNER
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
from utils import freeze_all_layers_except_last, get_device


# Components of training
device = get_device()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml")
    parser.add_argument('--log_dir', type=str, default='./models/')
    parser.add_argument('--compile_model', type=bool, default=False)
    args = parser.parse_args()
    
    config = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir

    model_config = GLiNERConfig(**vars(config))

    with open(config.train_data, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]

    print('Dataset is split...')

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    model_config.class_token_index = len(tokenizer)
    tokenizer.add_tokens([model_config.ent_token, model_config.sep_token])
    model_config.vocab_size = len(tokenizer)
    
    words_splitter = WordsSplitter(model_config.words_splitter_type)

    train_dataset = GLiNERDataset(train_data, model_config, tokenizer, words_splitter)
    test_dataset = GLiNERDataset(test_data, model_config, tokenizer, words_splitter)

    data_collator = DataCollatorWithPadding(model_config)

    model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)
    model.resize_token_embeddings([model_config.ent_token, model_config.sep_token], 
                                  set_class_token_index=False,
                                  add_tokens_to_tokenizer=False)

    model.to(device)

    # Freeze all layers except the last one
    freeze_all_layers_except_last(model)

    if args.compile_model:
        torch.set_float32_matmul_precision('high')
        model.compile_for_training()

    training_args = TrainingArguments(
        output_dir=config.log_dir,
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,  
        max_grad_norm=config.max_grad_norm,
        max_steps=config.num_steps,
        evaluation_strategy="epoch",
        save_steps=config.eval_every,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=4,  
        use_cpu=False,
        report_to="none",
        gradient_accumulation_steps=2,  # Adjusted gradient accumulation
        fp16=True,  # Enable mixed precision training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
