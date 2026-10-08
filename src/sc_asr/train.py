"""LoRA fine-tuning of Whisper on prepared JSONL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor


class AudioTextDataset(Dataset):
    def __init__(self, manifest: Path, processor: WhisperProcessor):
        self.items = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.base = manifest.parent
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        audio_path = Path(item["audio"])
        if not audio_path.is_absolute():
            audio_path = self.base / audio_path
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"{audio_path}: expected 16 kHz, found {sr}")
        features = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features[0]
        labels = self.processor.tokenizer(item["text"], return_tensors="pt", truncation=True,
                                          max_length=448).input_ids[0]
        return {"input_features": features, "labels": labels}


class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        inputs = torch.stack([f["input_features"] for f in features])
        labels = self.processor.tokenizer.pad({"input_ids": [f["labels"].tolist() for f in features]},
                                               return_tensors="pt")
        label_ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (label_ids[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            label_ids = label_ids[:, 1:]
        return {"input_features": inputs, "labels": label_ids}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="openai/whisper-tiny")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")


def train(args: argparse.Namespace) -> None:
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=False,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=AudioTextDataset(Path(args.train), processor),
        eval_dataset=AudioTextDataset(Path(args.validation), processor),
        data_collator=DataCollator(processor),
        processing_class=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
