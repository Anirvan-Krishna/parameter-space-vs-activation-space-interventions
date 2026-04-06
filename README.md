# Safety Alignment in LLMs: Parameter-Space vs. Activation-Space Interventions

This repository investigates two primary families of zero-training inference interventions designed to mitigate the vulnerability of instruction-tuned language models to jailbreak attempts. The project evaluates and contrasts methods that directly modify model weights with methods that alter internal representations dynamically at inference time.

## Overview of Interventions

* **Parameter-Space Interventions (RESTA):** This approach directly modifies the model's weights through task vector arithmetic and weight sparsification techniques.
* **Activation-Space Interventions (Function Vectors):** This method alters internal representations during inference by isolating and injecting task-specific hidden states.

## Models and Datasets

* **Base Model:** The foundational model used for all experiments is `Qwen/Qwen2.5-1.5B-Instruct`.
* **Judge Model:** `Qwen/Qwen2.5-7B-Instruct`, loaded in bfloat16 without quantization, is utilized for safety evaluations.
* **Helpful Domain (SFT):** The `medalpaca/medical_meadow_medqa` dataset is used for instruction tuning and evaluating downstream utility performance.
* **Harmful Prompts:** The `unalignment/toxic-dpo-v0.2` dataset is used to construct the harmful direction and to build few-shot prompts for the Function Vector task.
* **Safety Evaluation:** The final safety benchmark utilizes all 550 curated harmful queries from the `SoftMINER-Group/HarmEval` dataset.

## Methodology

### Part 1: Supervised Fine-Tuning (SFT) and DARE
The base model is fine-tuned on the medical Q&A dataset using LoRA. Only the attention projection matrices (`q_proj`, `k_proj`, `v_proj`) are adapted, while the MLP and embedding layers remain frozen. The Drop And REscale (DARE) algorithm is subsequently applied to sparsify the task delta vector using Bernoulli dropout and to rescale the surviving weights, implemented via the `mergekit` library.

### Part 2: Parameter-Space Safety Vector (RESTA)
To establish the harmful direction, the base model is fine-tuned on toxic prompts. The safety vector is calculated as the difference between the safe base direction and the harmful model direction. Using a linear merge, this safety vector is added to both the standard SFT model and the DARE-processed model.

### Part 3: Activation-Space Safety Vector (Function Vector)
Influential attention heads are identified by computing the Causal Indirect Effect (CIE). Toxic prompt embeddings are normalized and clustered using k-Means to generate structurally coherent clean and corrupted few-shot prompts. The Function Vector is constructed by summing the cached mean clean activations of the top-10 heads possessing the highest Average Indirect Effect (AIE). This vector is injected directly into the residual stream at the final token position using a custom PyTorch forward hook placed at layer 9.

## Evaluation and Comparative Analysis

Seven distinct model configurations were evaluated for safety, measured by an Unsafe Score, and utility, measured by ROUGE-L, METEOR, and BLEU metrics.

* **Utility Preservation:** Activation-space interventions demonstrated clear superiority in utility preservation. The Function Vector maintained the downstream capabilities acquired during supervised fine-tuning perfectly, yielding a BLEU score of 1.32. Conversely, the parameter-space RESTA intervention experienced catastrophic utility degradation in the target medical domain, dropping the BLEU score to 0.87.
* **Safety Enforcement:** RESTA proved slightly more effective at strictly enforcing safety constraints, achieving an Unsafe Score of 49.82%, compared to the Function Vector variant's 50.91%.
* **Conclusion:** Dynamic inference-time steering through Function Vectors offers a vastly superior balance compared to parameter-space arithmetic, successfully preserving task utility while delivering competitive safety alignment. Furthermore, the discrepancy in behavior when injecting vectors into DARE-processed models confirms that activation-space steering is highly sensitive to the underlying parameter distribution, requiring precise recalibration whenever base weights are altered.
