cross valid error > training error: variance problem
baseline to training error > then training to cv, then bias problem.

learning alg has high bias, then getting more training data would not help much, but if high variance, then might work (with small lambda regularization)

Get more training examples - fix high variance
try smaller sets of features - fix high variance
try additional features - fixes high bias
try adding polynomial features - fixes high bias
increase (fix high variance), decrease (fix high bias) lambda in regular.

bigger NN with good regular. is good!
large NN is low bias machine

Transfer learning: use parameters from previously trained model on something else, keep the initial params of first few layers
and eliminate the last layer and train on new data like that

precision = true positive / predicted positives
recall = true positive / actual positives

K means algorithm:
