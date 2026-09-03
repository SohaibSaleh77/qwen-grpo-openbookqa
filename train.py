import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

dataset_id = "allenai/openbookqa"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a science assistant. You will be given a fact and a multiple-choice question. "
    "Think step-by-step inside <think>...</think> tags. "
    "After your reasoning, output your final choice (A, B, C, or D) inside <answer>...</answer> tags. "
    "Do not output any text outside of these tags. "
    "Example:\n<think>The fact states X, which directly supports option Y.</think>\n<answer>Y</answer>"
)

def make_conversation(example):
    question = example['question_stem']
    # 'fact1' exists in the 'main' split of openbookqa
    fact = example.get('fact1', '') 
    choices = example['choices']
    
    # Format choices: A) text1 \n B) text2 ...
    labels = choices['label']  # ['A', 'B', 'C', 'D']
    texts = choices['text']
    choices_str = "\n".join([f"{l}) {t}" for l, t in zip(labels, texts)])
    
    user_content = f"Fact: {fact}\nQuestion: {question}\nChoices:\n{choices_str}"
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "gold_answer": example['answerKey']
    }

# --- Dense Reward Functions ---

def format_reward(completions, **kwargs):
    """Step 1: Reward strict format adherence to cure cold-start."""
    rewards = []
    # Enforces exact structure: <think>...</think>\n<answer>...</answer>
    pattern = re.compile(r"^<think>[\s\S]*<\/think>\n<answer>[\s\S]*<\/answer>$")
    for completion in completions:
        content = completion[0]["content"].strip()
        if pattern.match(content):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards

def valid_choice_reward(completions, **kwargs):
    """Step 2: Reward choosing a valid letter (A, B, C, D)."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            ans = match.group(1).strip().upper()
            if ans in ['A', 'B', 'C', 'D']:
                rewards.append(0.2)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward(completions, gold_answer, **kwargs):
    """Step 3: Reward correct answer."""
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        content = completion[0]["content"]
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            ans = match.group(1).strip().upper()
            if ans == gold:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Science QA")
    
    # Dataset arguments
    parser.add_argument("--train_split_percentage", type=int, default=30, help="Percentage of training data to use")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="Qwen-GRPO-OpenBookQA", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--generation_batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--bf16", type=bool, default=True, help="Use bfloat16 precision")
    parser.add_argument("--max_completion_length", type=int, default=384, help="Maximum completion length")
    parser.add_argument("--num_generations", type=int, default=8, help="Number of generations per prompt")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging steps")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load dataset ('main' split contains the fact1 column)
    train_percentage = f"train[:{args.train_split_percentage}%]"
    train_dataset = load_dataset(dataset_id, "main", split=train_percentage)
    train_dataset = train_dataset.map(make_conversation)
    
    # Load Model and Tokenizer explicitly
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # LoRA config tailored for Qwen architecture
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        save_strategy="no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        report_to=["tensorboard"],
        logging_steps=args.logging_steps,
        beta=args.beta,
        temperature=args.temperature,
    )
    
    trainer = GRPOTrainer(
     
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, valid_choice_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset
    )
    
    trainer.train()

if __name__ == "__main__":
    main()