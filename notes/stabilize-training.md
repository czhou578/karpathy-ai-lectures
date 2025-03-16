# How to stabilize LLM Training

## Cosine Decay and Warmup for Learning Rate

- Warm-up: Allows the model to initially explore the parameter space without being overly sensitive to noisy gradients.

- Decay: Gradually reduces the learning rate as training progresses, allowing the model to converge to a more precise minimum.

Common schedules: Cosine annealing, linear decay, cyclical learning rates.

## Lower learning rate

- If training is unstable, reduce it (e.g., 5e-5 or 3e-5).

## Gradient Clipping

- What it does: Limits the maximum norm of the gradients during backpropagation.
- Why it helps: Prevents gradient explosions, which can cause sudden jumps in loss and destabilize training.

## Ensure LayerNorm Before Attention & FFN

- Many transformer models place LayerNorm before self-attention and the feedforward block, instead of after.

## Weight decay

- optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

## Check for NaN in gradients

- torch.isinf(model.parameters()).any()

## Gradient Accumulation

- What it does: Accumulates gradients over multiple mini-batches before performing an optimizer step.

- Why it helps: Allows you to simulate larger batch sizes, which can improve stability and reduce variance in the gradients.

- How to implement: Accumulate gradients in a loop and then call optimizer.step() after a certain number of steps.
