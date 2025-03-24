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

## Parallelize Data Loading

```
    from torch.utils.data import TensorDataset, DataLoader

    # Create tensor datasets
    train_dataset = TensorDataset(training_data[:-block_size], training_data[1:1+block_size])
    val_dataset = TensorDataset(validation_data[:-block_size], validation_data[1:1+block_size])

    # Create data loaders with multiple workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
```

The Dataset is ab abstraction to be able to load and process each sample of your dataset lazily, while the DataLoader takes care of shuffling/sampling/weigthed sampling, batching, using multiprocessing to load the data, use pinned memory etc.
