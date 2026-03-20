#!/usr/bin/env python3
"""Reproducible fine-tuning script for Meta-Llama-3-8B-Instruct on ESConv."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import (
    DEFAULT_FINETUNE_OUTPUT_DIR,
    ensure_parent_dir,
    init_env,
    path_from_env,
)

TRAINING_PROMPT_TEMPLATE = """Emotional Support Agent.
Your task is to generate emotional-support responses that are warm,
empathetic, and concise, while respecting strict role constraints.
Role Constraints:
- You are a compassionate AI companion for emotional support.
- You are NOT a human -- never claim personal experiences.
- You are a supportive friend, NOT a clinician.
Few-shot Demonstrations:
Example 1: User: {val_example_1_user}
Assistant: {val_example_1_assistant}
Example 2: User: {val_example_2_user}
Assistant: {val_example_2_assistant}
Current User Message:
{user_message}
Task Description:
Given the user's message, generate a warm, empathetic response that:
1) Acknowledges the user's emotions and validates their feelings.
2) Asks clarifying questions when appropriate.
3) Offers gentle support and practical next steps when appropriate.
4) Avoids clinical language and medical/therapeutic claims.
5) Stays conversational and concise.
Answer:"""


def parse_args() -> argparse.Namespace:
    init_env()
    finetune_default = path_from_env("PERSONARAG_FINETUNE_OUTPUT_DIR", DEFAULT_FINETUNE_OUTPUT_DIR)

    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-3-8B-Instruct on ESConv")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset-name", default="thu-coai/esconv")
    parser.add_argument("--output-dir", type=Path, default=finetune_default / f"llama3-esconv-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--val-example-1-user", default="I've been feeling really anxious about my exams.")
    parser.add_argument("--val-example-1-assistant", default="That sounds really heavy. It makes sense to feel anxious under so much pressure. Would it help to talk through what feels most overwhelming right now?")
    parser.add_argument("--val-example-2-user", default="My friend stopped replying and I feel rejected.")
    parser.add_argument("--val-example-2-assistant", default="I'm sorry you're going through that. Being left without answers can hurt a lot. Do you want to share what this friendship means to you?")
    parser.add_argument("--use-expanding-window", action="store_true", default=True)
    return parser.parse_args()


def build_training_user_prompt(args: argparse.Namespace, user_message: str) -> str:
    return TRAINING_PROMPT_TEMPLATE.format(
        val_example_1_user=args.val_example_1_user,
        val_example_1_assistant=args.val_example_1_assistant,
        val_example_2_user=args.val_example_2_user,
        val_example_2_assistant=args.val_example_2_assistant,
        user_message=user_message,
    )


def format_one_dialogue_to_examples(example: dict, args: argparse.Namespace) -> list[str]:
    try:
        payload = json.loads(example["text"])
    except Exception:
        return []

    dialog = payload.get("dialog", [])
    if not dialog:
        return []

    examples: list[str] = []
    history: list[tuple[str, str]] = []
    pending_user: list[str] = []

    for turn in dialog:
        speaker = turn.get("speaker", "")
        text = (turn.get("text", "") or "").strip()
        if not text:
            continue

        if speaker == "usr":
            pending_user.append(text)
            history.append((speaker, text))
            continue

        if speaker != "sys" or not pending_user:
            history.append((speaker, text))
            continue

        current_user_message = " ".join(pending_user)
        prompt_user = build_training_user_prompt(args, current_user_message)

        if args.use_expanding_window:
            history_context = []
            for h_speaker, h_text in history[:-1]:
                role = "user" if h_speaker == "usr" else "assistant"
                history_context.append(f"[{role}] {h_text}")
            if history_context:
                prompt_user = f"Conversation History:\n" + "\n".join(history_context) + "\n\n" + prompt_user

        formatted = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a compassionate AI companion for emotional support.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt_user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
        )
        examples.append(formatted)
        pending_user = []
        history.append((speaker, text))

    return examples


def build_training_dataset(raw_split, args: argparse.Namespace) -> Dataset:
    texts: list[str] = []
    for sample in raw_split:
        texts.extend(format_one_dialogue_to_examples(sample, args))
    return Dataset.from_dict({"text": texts})


def main() -> None:
    args = parse_args()

    ensure_parent_dir(args.output_dir / "placeholder.txt")

    print("=" * 70)
    print("REPRODUCIBLE FINE-TUNING: LLAMA-3-8B + ESCONV")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {args.output_dir}")

    print("\nLoading dataset...")
    ds = load_dataset(args.dataset_name)

    print("Building training examples...")
    train_ds = build_training_dataset(ds["train"], args)
    val_ds = build_training_dataset(ds["validation"], args)
    test_ds = build_training_dataset(ds["test"], args)

    _ = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    print(f"Train examples: {len(train_ds)}")
    print(f"Validation examples: {len(val_ds)}")
    print(f"Test examples: {len(test_ds)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading model/tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Preparing LoRA...")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        save_total_limit=3,
        packing=False,
        dataloader_pin_memory=False,
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        args=train_cfg,
    )

    print("\nStarting training...")
    trainer.train()

    print("Saving artifacts...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    (args.output_dir / "training_prompt_template.txt").write_text(TRAINING_PROMPT_TEMPLATE, encoding="utf-8")
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "dataset": args.dataset_name,
                "num_train_epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "use_expanding_window": args.use_expanding_window,
                "train_examples": len(train_ds),
                "validation_examples": len(val_ds),
                "test_examples": len(test_ds),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
