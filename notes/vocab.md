## KV Cache: avoiding recomputation by using cached key and value matrices in transformer architectures

## Hyperparameter: choices about algorithm that you set rather then learn

## Quantization: representing values in fewer bits then FP32

    - 8 bit or 4 bit

example: (1000 _ 128) matrix which will contain 1000 _ 128 _ 4 bytes in FP32
INT8 requires 1 byte per value so result will be 1000 _ 128 bytes

Can be used to deploy large models on edge devices like it is in huggingface

## Dimensionality Reduction: reduces features / columns in key val matrices

example: (1000 _ 128) matrix downscaled to (1000 _ 32) will obviously save memory

Can be used in speech to text models to filter out redundant features in long sequences

## Sparse Representation: store sig figs in a matrix. Zero out or remove useless values

    - Top K Sparsity: Retain only the top-k largest values in each row of the key or value matrix.
    - Threshold Sparsity: Zero out values below a certain threshold.

example: [0.01, 0.50, 0.03, 0.70, 0.02]

Top K with K = 2: [0, 0.50, 0, 0.70, 0]
Threshold Sparsity = 0.05: [0, 0.50, 0, 0.70, 0]

Sparse representations are useful in recommender systems and memory-intensive applications like GPT models where storing and processing every value can be prohibitive.

## Memory Mapping / Chunking: Instead of storing the entire KV cache in memory, chunk the data into smaller parts and load only the required chunks during computation.

example: (10000, 64), divide into 10 chunks of 1000 and store only one chunk in memory at one time, reduce peak memory usage

Streaming models where inputs are processed incrementally, such as in real-time transcription or language translation.

# Mixture of Experts

### Experts: sub-networks (could be 100s of them)

### Gating network: determines which experts to activate for a given input. It outputs a probability distribution over the experts, often selecting the top-k highest probabilities.

- can be deterministic or probabilistic
- Ensuring all experts are used equitably, preventing overloading of certain experts while others remain underutilized.
- Efficiently implementing sparse activation to avoid computation for unselected experts.

example: 10 experts and gating network that activates 2 experts per input. Input x predicts which 2 experts are most relevant for x.

Only these 2 experts process x and their outputs are combined

### Sparsity in MoE

- Efficiency: Activating only a few experts reduces the number of parameters and operations involved in each forward pass

- Also very scalable. Can add more experts without increasing cost.

### Specialization of Experts

- Each expert in an MoE can specialize in processing a specific type of input. During training, different experts are encouraged to focus on different regions of the input space. Allows processing of heterogenous data.

- In an MoE with 10 experts, if the gating network consistently activates only 2 experts for all inputs, the remaining 8 experts are effectively unused. A load balancing term in the loss function penalizes this behavior, encouraging the gating network to distribute activations more evenly.

## Regularization improves model generalization by preventing overfitting

### Adding constraints to model complexity

### Encouraging the model to focus on the most relevant patterns in the data.

L2 Regularization:

It penalizes large weights by adding the sum of the squared values of the weights to the loss function. The regularization term encourages the model to keep weights smaller, effectively reducing the model's complexity.

## Dropout: a fraction of neurons in a layer is randomly "dropped" (set to zero) during each training iteration. This prevents the network from becoming overly reliant on specific neurons, encouraging more robust feature representations.

## Early Stopping: halts training once the model's performance on a validation set stops improving, preventing overfitting and saving computational resources.

## Data Augmentation: generates additional training samples by applying transformations to the existing data. This increases the dataset's diversity and reduces overfitting.

## Weight Sharing: reuses the same set of weights across multiple parts of the model, reducing the number of parameters and improving generalization.

- Reduces the model's size and computational complexity.
