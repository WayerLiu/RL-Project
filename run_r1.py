# run_r1.py

import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "tau/commonsense_qa"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

########################
# Helper functions
########################

def format_reward_func(completions, **kwargs):
    """
    Checks if the output has <think>...</think>\n<answer>...</answer> format.
    """
    rewards = []
    for completion in completions:
        try:
            completion = "<think>" + completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, answerKey, **kwargs):
    """
    Extracts <answer>...</answer> and checks if it matches the ground-truth label.
    """
    rewards = []
    for completion, gt in zip(completions, answerKey):
        try:
            completion = "<think>" + completion
            match = re.search(r"<answer>\s*([A-E])\s*</answer>", completion)
            if match and match.group(1).strip().upper() == gt.strip().upper():
                rewards.append(1.0)
                if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def build_r1_prompt(example, tokenizer):
    """
    Builds the R1-style prompt for CommonsenseQA.
    """
    question = example["question"]
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]
    choices_str = "\n".join([f"{label}: {text}" for label, text in zip(labels, choices)])

    r1_prefix = [
        {"role": "system", "content": "You are a helpful assistant. You first think through the reasoning, then provide the correct answer clearly in an <answer>...</answer> tag."},
        {"role": "user", "content": f"Given question: {question}, choose from these options:\n{choices_str}\nShow your thoughts in <think> </think> tags. Return the final answer option in <answer> </answer> tags, e.g., <answer>A</answer>."},
        {"role": "assistant", "content": "Let me solve this question.\n<think>"}
    ]
    return tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True)

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

########################
# Main training function
########################

def grpo_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Training parameters: {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # dataset = dataset.shuffle(seed=42).select(range(100))  # 小规模测试

    # Process dataset into prompts
    dataset = dataset.map(lambda x: {
        "prompt": build_r1_prompt(x, tokenizer)
    })

    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer = GRPOTrainer(
    model=model_args.model_name_or_path,
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=get_peft_config(model_args),
)


    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    logger.info(f"*** Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete. Saving model... ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "commonsenseqa", "r1-prompt"]})

    if training_args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()

    logger.info("*** All done! ***")

########################
# Main entry
########################

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    grpo_function(model_args, script_args, training_args)

if __name__ == "__main__":
    main()