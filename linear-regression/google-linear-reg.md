When choosing the best loss function, consider how you want the model to treat outliers

For instance, MSE moves the model more toward the outliers,

## Hyperparameters:

- epochs
- batch size: to the number of examples the model processes before updating its weights and bias.
- learning rate: floating point number you set that influences how quickly the model converges.

The learning rate determines the magnitude of the changes to make to the weights and bias during each step of the gradient descent process. The model multiplies the gradient by the learning rate to determine the model's parameters

A learning rate that's too large never converges because each iteration either causes the loss to bounce around or continually increase.

Stochastic Gradient Descent - only uses batch size of 1 per iteration.
Compromise: Mini batch SGD: For N number of data points, the batch size can be any number greater than 1 and less than N. The model chooses the examples included in each batch at random, averages their gradients, and then updates the weights and bias once per iteration.

Sigmoid function: 1 / (1 + e ^ -z) where z = bias + w1x1 + w2x2 + ....wnxn

Loss function for logistic regression is Log Loss.

## Regularization, a mechanism for penalizing model complexity during training

# Classification

- Accuracy: proportion of all classifications that were correct, whether positive or negative.

- The true positive rate (TPR), or the proportion of all actual positives that were classified correctly as positives, is also known as recall.

- The false positive rate (FPR) is the proportion of all actual negatives that were classified incorrectly as positives, also known as the probability of false alarm. Perfect
  model would have zero false positives.

- Precision is the proportion of all the model's positive classifications that are actually positive.

- Precision improves as false positives decrease, while recall improves when false negatives decrease. But as seen in the previous section, increasing the classification threshold tends to decrease the number of false positives and increase the number of false negatives, while decreasing the threshold has the opposite effects. As a result, precision and recall often show an inverse relationship, where improving one of them worsens the other.

## Numerical Data

However, feature vectors seldom use the dataset's raw values.

- Normalization: Converting numerical values into a standard range.
- Binning (also referred to as bucketing): Converting numerical values into buckets of ranges.

### Normalization

- Feature X spans the range 154 to 24,917,482, Feature Y spans the range 5 to 22. Normalizing puts them at around the same range
- Helps models converge more quickly. Gradient descent can bounce and slow convergence

- Helps prevent NaN since if feature values are too high and exceed floating point limit

### Linear Scaling

- converting floating-point values from their natural range into a standard range—usually 0 to 1 or -1 to +1.

Works when: The feature contains few or no outliers, and those outliers aren't extreme, lower + upper bounds don't change through time, uniformly distributed

### Z-score Scaling

Representing a feature with Z-score scaling means storing that feature's Z-score in the feature vector. For example, the following figure shows two histograms:
Works when the data follows a normal distribution or a distribution somewhat like a normal distribution.

### Log Scaling

Works when the data conforms to a power law distribution. Casually speaking, a power law distribution looks as follows:
Low values of X have very high values of Y.
As the values of X increase, the values of Y quickly decrease. Consequently, high values of X have very low values of Y.

### Clipping:

Clipping is a technique to minimize the influence of extreme outliers. In brief, clipping usually caps (reduces) the value of outliers to a specific maximum value. Clipping is a strange idea, and yet, it can be very effective.

## Binning

Binning is a good alternative to scaling or clipping when either of the following conditions is met:

The overall linear relationship between the feature and the label is weak or nonexistent.
When the feature values are clustered.

## Feature Crosses

Feature crosses are created by crossing (taking the Cartesian product of) two or more categorical or bucketed features of the dataset. Like polynomial transforms, feature crosses allow linear models to handle nonlinearities.

Training, validation, test set: train, do initial testing as training happens, then officially test

A loss curve plots a model's loss against the number of training iterations. A graph that shows two or more loss curves is called a generalization curve.

Learning rate and regularization rate tend to pull weights in opposite directions. A high learning rate often pulls weights away from zero; a high regularization rate pulls weights towards zero.

If the regularization rate is high with respect to the learning rate, the weak weights tend to produce a model that makes poor predictions. Conversely, if the learning rate is high with respect to the regularization rate, the strong weights tend to produce an overfit model.

## Understanding Loss Curves:

Case 1: Oscillating loss curve

- reduce the learning rate
- reduce training set to tiny number of trustworthy examples
- check data against data schema and remove the bad examples

Case 2: Loss curve with a sharp jump

- The input data contains one or more NaNs—for example, a value caused by a division by zero.
- The input data contains a burst of outliers.

Case 3: Loss curve gets stuck

- The training set contains repetitive sequences of examples.

# Common Activation Functions

## Sigmoid

- 1 / 1 + e^x

## RELU

- The rectified linear unit activation function (or ReLU, for short) transforms output using the following algorithm:

If the input value is less than 0, return 0.
If the input value is greater than or equal to 0, return the input value.

- Doesn't suffer from vanishing gradient problem during training.

## Exploding Gradients

- If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms. In this case you can have exploding gradients: gradients that get too large to converge. Batch normalization can help prevent exploding gradients, as can lowering the learning rate.

## Dead ReLU

- caused by bad weight updates, large negative weights, high learning rate, vanishing gradient

- Using leakyRelu introduces a small slope (𝛼) for negative inputs to keep neurons alive:
- Parametric Relu with learnable (𝛼) which adapts during training
- batch norm
- ELU, and GELU, preventing neurons from dying
- He initialization

## Softmax

- For one-vs.-all, we applied the sigmoid activation function to each output node independently, which resulted in an output value between 0 and 1 for each node, but did not guarantee that these values summed to exactly 1.

For one-vs.-one, we can instead apply a function called softmax, which assigns decimal probabilities to each class in a multi-class problem such that all probabilities add up to 1.0. This additional constraint helps training converge more quickly than it otherwise would.

Full softmax vs Candidate Sampling

- Full softmax is fairly cheap when the number of classes is small but becomes prohibitively expensive when the number of classes climbs. Candidate sampling can improve efficiency in problems having a large number of classes.

In general, you can create a hidden layer of size d in your neural network that is designated as the embedding layer, where d represents both the number of nodes in the hidden layer and the number of dimensions in the embedding space.

## Recurrent Neural Networks

- a type of neural network that trains on a sequence of tokens.

Ex: can gradually learn (and learn to ignore) selected context from each word in a sentence, kind of like you would when listening to someone speak.

long contexts is constrained by the vanishing gradient problem.

## Distillation

- Distillation creates a smaller version of an LLM. The distilled LLM generates predictions much faster and requires fewer computational and environmental resources than the full LLM.

### Offline inference

- In other words, rather than responding to queries at serving time, the trained model makes predictions in advance and then caches those predictions.

Static vs dynamic inference

- model makes predictions on a bunch of common unlabeled examples and then caches those predictions somewhere vs only makes predictions on demand

## training-serving skew

- Training-serving skew is more dangerous when your system performs dynamic (online) inference.

Full machine learning system requires:

Validating input data.
Validating feature engineering.
Validating the quality of new model versions.
Validating serving infrastructure.
Testing integration between pipeline components.

Check that components work together by writing an integration test that runs the entire pipeline end-to-end.

If your dataset has one or more features that have missing values for a large number of examples, that could be an indicator that certain key characteristics of your dataset are under-represented.
