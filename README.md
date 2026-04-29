# Neural Network from Scratch

A deep learning framework implemented from scratch in C++ using the Eigen3 library. I developed this project to get a solid understanding of how neural networks learn via gradient descent.

## Features

- **Models**: Multi-Layer Perceptron (MLP)
- **Layers**: Dense and activation layers (ReLU, Softmax, etc.)
- **Loss Functions**: Cross-entropy, Mean Squared Error

## Example
The framework has been tested with the MNIST dataset for handwritten digit classification.
See `main.cpp` for an example training loop. Below is the output for this example :

```text
========================================
MNIST Digit Classification
========================================

Loading MNIST training data...
Loaded 500 training samples
Loaded 100 test samples

Training MLP for digit classification...
Learning Rate: 0.05, Epochs: 25
Epoch           Avg CrossEntropy        Accuracy
-----           --------------  --------
0               1.239985        12.000000%
1               1.179416        18.799999%
2               1.165002        18.799999%
3               1.149304        21.000000%
4               0.968361        29.200001%
5               0.884408        37.400002%
6               0.873769        38.599998%
7               0.829362        40.400002%
8               0.764404        45.200001%
9               0.731804        47.400002%
10              0.667728        52.000000%
11              0.636701        53.799999%
12              0.596348        56.599998%
13              0.592217        58.200001%
14              0.608679        55.400002%
15              0.534386        61.599998%
16              0.544500        61.000000%
17              0.541801        61.000000%
18              0.498282        64.199997%
19              0.482181        66.400002%
20              0.504707        65.800003%
21              0.523057        62.599998%
22              0.500750        66.400002%
23              0.479129        67.800003%
24              0.484345        66.599998%

========================================
Testing on Test Set
========================================
Sample 0: Predicted = 7, Actual = 7 [CORRECT]
Sample 1: Predicted = 2, Actual = 2 [CORRECT]
Sample 2: Predicted = 1, Actual = 1 [CORRECT]
Sample 3: Predicted = 0, Actual = 0 [CORRECT]
Sample 4: Predicted = 7, Actual = 4 [WRONG]
Sample 5: Predicted = 1, Actual = 1 [CORRECT]
Sample 6: Predicted = 3, Actual = 4 [WRONG]
Sample 7: Predicted = 3, Actual = 9 [WRONG]
Sample 8: Predicted = 2, Actual = 5 [WRONG]
Sample 9: Predicted = 7, Actual = 9 [WRONG]

Test Accuracy: 58.000000% (58/100)
Test Avg CrossEntropy: 0.566316
```
