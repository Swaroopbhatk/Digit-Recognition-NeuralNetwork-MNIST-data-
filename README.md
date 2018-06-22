<b><i>Date: 25-02-2018</i></b>   
<b><i>Name: Swaroop S Bhat</i></b>

# Implementation of Neural Network and Optimization

To understand working of neural network let us start with a sample of simple neural network model with 1 hidden layer using moons400.csv dataset.  
This dataset is having 2 features and binary class label(0 and 1).

##### Algorithm for 1 hidden layer neural network with structured dataset

    1. We will convert the labels through one hot encoding into a format which network can understand. But this step is optional. Not required for simple and numeric labels.
    2. Repeat step 3 to 5 for training set until the end of epoch/iterations
    3. Divide the dataset:
        + 280 instances in training set.
        + 120 instances in testing set.
    4. Using feedforward get sigmoid values for outer layer and hidden layer with initial values of weights and biases
    5. Apply back propagation to get gradients which will update the weights and biases.
    6. After the end of epoch, use the updated weights to predict for test set.
    7. Plot Learning Curves, ROC and Confusion matrix to check performance.

##### Scatter Plot of Dataset moons.csv:

![moonsscatter](https://user-images.githubusercontent.com/32418025/41792836-1cfe46be-7652-11e8-80e3-580edf309af4.png)

##### Sigmoid Activation Function

The main reason why we use sigmoid function is because it exists between (0 to 1). Therefore, it is especially used for models where we have to predict the probability as an output.Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice.
Ref: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

##### Simulation Results:

![gdmoons](https://user-images.githubusercontent.com/32418025/41792947-760521b0-7652-11e8-86c5-0c0fd386b6be.png)
