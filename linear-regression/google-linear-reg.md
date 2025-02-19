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
