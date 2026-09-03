
# Dense GRPO Training: OpenBookQA

## 📚 Educational Project
This project explores **Group Relative Policy Optimization (GRPO)** using a dense reward system to teach a 0.5B model how to reason and answer multiple-choice science questions.

## 🎯 The Dense Reward Pipeline
Instead of a single binary reward, the model is scored on a gradient of success:
1. **Format:** Did it use `<think>` and `<answer>` tags? (0.1 points)
2. **Valid Choice:** Did it extract a valid letter (A, B, C, D)? (0.2 points)
3. **Accuracy:** Was the chosen letter correct? (1.0 points)

## 🚀 Run It
```bash
pip install torch transformers datasets peft trl accelerate
python train.py

