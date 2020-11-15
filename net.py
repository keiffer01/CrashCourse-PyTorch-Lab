import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        '''
        You will build your own neural network here!
        Look back at the slides for some inspiration!

        Recap of Layers:
        - nn.Linear(in, out):
            a fully-connected layer that something of size "in" and returns
            something of size "out"
        - nn.Conv2d(in_channels, out_channels, kernel_size):
            convolutional layer that works on 2d images with number of channels
            given by "in_channels". Applies a number of filters given by
            "out_channels", with a kernel_size that can be an integer if square,
            or a tuple.

        Some Activation functions:
        - nn.Sigmoid(): applies sigmoid activation function (maps to range (0, 1))
        - nn.ReLU(): applies Rectified Linear Unit function (maps to (0, max(input)))
        - nn.MaxPool2d(kernel_size): Applied after Conv2d with the given kernel size
        '''
        ### YOUR CODE HERE ###



    def forward(self, x):
        '''
        You will define the feedforward function here! Make use of the layers
        and activation functions you definied in __init__.
        '''
        ### YOUR CODE HERE ###
