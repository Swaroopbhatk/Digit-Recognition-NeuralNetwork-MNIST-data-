<b><i>Date: 25-02-2018</i></b>   
<b><i>Name: Swaroop S Bhat</i></b>

## Final Resuts - Comparision of optimization techniques:
    - Batch Gradient Descent
    - Mini Batch Gradient Descent
    - Adam Optimization with mini batch GD
    - ReLU activation with adam optimization

![image](https://user-images.githubusercontent.com/32418025/41793570-763be310-7654-11e8-8a76-f3f5c751e319.png)
![image](https://user-images.githubusercontent.com/32418025/41793545-615c4f02-7654-11e8-809d-a9ff46ecabbc.png)

Further details is given below.

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


## MNIST Digit Recognition with 3 and 9 (works for 1 or 2 hidden layers):

<b>We will use subset of original dataset for our neural network mdoel. 7000 training data and 1000 testing data.</b>

Note: All below mentioned optimizations is created in seperate codes to increase the readabilty.
  
<b>Optimisation Used:</b>
<li>Implemented 2 hidden layers. Code runs for both 1 and 2 hidden layers and any number of nodes in hidden layer.</li>
<li>Batch Gradient Descent with sigmoid activation.</li>
<li>Mini Batch Gradient Descent with sigmoid activation.</li>
<li>Adam Optimisation with sigmoid activation.</li>
<li>Adam Optimisation with ReLU activation.</li>
<li>Principal Component Analysis (PCA) for feature reduction.</li>
<li>Model Comparisions.</li>

### Data Preprocessing
    - Unzipped MNIST data is stored in folder called samples in home directory.
    - Loaded the testing and training data using MNIST library.
    - Laoding training features and corresponding labels.
    - Loading Testing features and corresponding labels.
    - Randomly shuffling the data and Normalising the training features (max normalisation from sckit learn package).
    - Converting training labels and test labels to binary (0 for label 3 and 1 for label 9).
    - Conterting training and testing data to matrix or numpy form.

### Optimization 1 - Batch Gradient Descent

    Batch gradient descent computes the gradient using the whole dataset. 
    Batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset:
    θ=θ−η⋅∇J(θ).
    As we need to calculate the gradients for the whole dataset to perform just one update, batch gradient descent can be very slow and is intractable for datasets that don't fit in memory.    
    
#### Algorithm
    1. Initialize Weights and Biases
    2. For each iteration in epoch:
        *  Consider entire training features as an input. 
        *  Feedforward: Get activation values using sigmoid function for each layer.
        *   Back Propagation: Get error in output layer and calculate derivative of error in each hidden layers.
                    -  Gradients: use these gradients to update previous value of weights and biases.
    3. Use the updated weights and biases in feed forward to predict labels for test set. 
    
#### Simulation Results:

![gd1](https://user-images.githubusercontent.com/32418025/41793060-ded163ca-7652-11e8-8468-79bd83643d14.png)
![image](https://user-images.githubusercontent.com/32418025/41793092-f8963d62-7652-11e8-8278-3f0c516186be.png)
![image](https://user-images.githubusercontent.com/32418025/41793168-25e58976-7653-11e8-8e02-6fed88dec14e.png)


### Optimization 2 - Mini Batch Gradient Descent
    This is similar to stochastic gradient descent, where set of batches are taken to update weights instead of single 
    instance of training set. Hence, it is faster than stochastic gradient descent and slower than gradient descent 
    (entire traing set is considered as one batch). As a result it is possible that mini batch gradient descent converges 
    faster than batch gradient descent.
    
    

#### Algorithm
    1. Initialize Weights and Biases.
        2. For each iteration in epoch:
            - Create Multiple batches here 7 batches are created.
            - For each iteration in batch:
                *  Consider the current batch. 
                *  Feedforward: Get activation values using sigmoid function for each layer.
                *  Back Propagation: Get error in output layer and calculate derivative of error in each hidden layers.
                    +  Gradients: use these gradients to update previous value of weights and biases.
                * Continue the above processs for all remaining batches and update weights.
        3. Use the updated weights and biases in feed forward to predict labels for test set. 

Ref: https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent

#### Simulation Results:

![image](https://user-images.githubusercontent.com/32418025/41793223-57368c5a-7653-11e8-90d2-8371a6c22c37.png)
![image](https://user-images.githubusercontent.com/32418025/41793274-766bcef0-7653-11e8-813b-14c3f626d3c7.png)
![image](https://user-images.githubusercontent.com/32418025/41793315-89b53afa-7653-11e8-9034-5dd5b82e88d2.png)


### Optimization 3 - Adam Optimization
    Adam is an update to the RMSProp optimizer. In this optimization algorithm, running averages of both the gradients and    the second moments of the gradients are used.
    
#### Algorithm
    1. Initialize Weights and Biases.
        2. For each iteration in epoch:
            - Create Multiple batches here 7 batches are created.
            - For each iteration in batch t:
                Initialize Vdw=0, Sdw=0, Vdb=0, Sdb=0, epsilon = 0.0000001, beta1 = 0.9, beta2 = 0.999
                *  Consider the current batch. 
                *  Feedforward: Get activation values using sigmoid function for each layer.
                *  Back Propagation: Get error in output layer and calculate derivative of error in each hidden layers.
                    - Compute dw, db using current mini batch
                    - Vdw = Beta1*Vdw+(1-Beta1)*dw; Vdb=Beta1*Vdb+(1-Beta1)*db
                    - Sdw = Beta2*Sdw+(1-Beta2)*dw^2; Vdb=Beta1*Sdb+(1-Beta1)*db^2
                    - Compute Vdw corrected = Vdw/(1-Beta1^t); Vdb corrected = Vdb/(1-Beta1^t)
                    - Compute Sdw corrected = Sdw/(1-Beta1^t); Sdb corrected = Sdb/(1-Beta1^t)
                * Update weights.
                    - W = W - alpha*(Vdw corrected/(sqrt(Sdw corrected)+epsilon))
                    - b = b - alpha*(Vdb corrected/(sqrt(Sdb corrected)+epsilon))
        3. Use the updated weights and biases in feed forward to predict labels for test set. 
   
<li>Ref:https://www.youtube.com/watch?v=JXQT_vxqwIs (Andrew Ng)</li>
<li>Ref:https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam (Wiki) </li>

#### Simulation Results:

![image](https://user-images.githubusercontent.com/32418025/41793396-cc77443c-7653-11e8-86bb-3afac23d5564.png)
![image](https://user-images.githubusercontent.com/32418025/41793419-e39ecf7c-7653-11e8-8222-e59a1072fc85.png)


### Optimization 4 - ReLU Activation with Adam Optimization
    Rectifier linear unit or its more widely known name as ReLU becomes popular for the past several years since its performance  and speed. In contrast to other common activation functions, ReLU is a linear function. In other words, its derivative is either 0 or 1.
    i.e.f(x) = max(x, 0)
    
<li>Ref: https://sefiks.com/2017/08/21/relu-as-neural-networks-activation-function/ </li>
<li>Ref: https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions </li>

#### Simulation Results:

![image](https://user-images.githubusercontent.com/32418025/41793471-18fb661c-7654-11e8-8014-745c4f286a84.png)
![image](https://user-images.githubusercontent.com/32418025/41793495-33d372e0-7654-11e8-95f7-95202f078d10.png)


### Comparision of optimization techniques:

![image](https://user-images.githubusercontent.com/32418025/41793545-615c4f02-7654-11e8-809d-a9ff46ecabbc.png)
![image](https://user-images.githubusercontent.com/32418025/41793570-763be310-7654-11e8-8a76-f3f5c751e319.png)


    
