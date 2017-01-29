# kaggle-digit-recognizer

## Results

```
Multilayer perceptron #2
classify_mlp_2.py

1 hidden layer, 800 neurons, batchnorm, relu, dropout 0.4
2 hidden layer, 800 neurons, batchnorm, relu, dropout 0.4
1 output layer, 10 neurons, softmax
Xavier normal initialization
Weight regularization: L2(0.00001)
Optimizer: Adam with defaults
Batch size: 64
Learning rate schedule: steps by epochs (see file)

282 epochs
Last epoch: 3s - loss: 0.0284 - acc: 0.9989
               - val_loss: 0.0832 - val_acc: 0.9876
Train time: ~15 minutes
Test: 0.98657
```

```
Multilayer perceptron #1 Mk II - less gerularized, iterator and functional API
classify_mlp_1.py

1 hidden layer, 1000 neurons, batchnorm, relu, dropout 0.4
1 output layer, 10 neurons, softmax
Xavier normal initialization
Weight regularization: L2(0.00001)
Optimizer: Adam with defaults
Batch size: 64
Learning rate schedule: steps by epochs (see file)

183 epochs
Last epoch: 4s - loss: 0.0173 - acc: 0.9994
               - val_loss: 0.0789 - val_acc: 0.9852
Train time: ~12.2 minutes
Test: 0.98557
```

```
Multilayer perceptron #1
classify_mlp_1.py

1 hidden layer, 1000 neurons, batchnorm, relu, dropout 0.4
1 output layer, 10 neurons, softmax
Xavier normal initialization
Weight regularization: L2(0.00005)
Optimizer: Adam with defaults
Batch size: 64
Learning rate schedule: steps by epochs (see file)

186 epochs
Last epoch: 2s - loss: 0.0245 - acc: 0.9989
               - val_loss: 0.0758 - val_acc: 0.9860
Train time: ~7 minutes (GTX 1060 6 Gb)
Test: 0.98443
```

```
SVM, RBF kernel
MinMaxScaler [0, 1]
PCA = 35_whitening
C = 1000, γ = 0.049
CV: [0.98334325  0.98417232  0.97904512  0.98273193  0.98237256]
Test: 0.98471
```

```
SVM, RBF kernel
MinMaxScaler [0, 1]
PCA = 35_whitening
C = 64, γ = 0.03125
CV: [0.98132064  0.98286326  0.97761638  0.98297011  0.98189614]
Test: 0.98357
```

```
kNN, k=3
MinMaxScaler [0, 1]
PCA=35_no_whitening
CV: [0.97572873  0.97607997  0.96928206  0.97153745  0.97451167]
Test: 0.97371
```
