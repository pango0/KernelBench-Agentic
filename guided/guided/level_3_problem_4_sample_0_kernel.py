import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConv2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None, padding=0, stride=1, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias, padding, stride, dilation, groups)
        output = F.conv2d(input, weight, bias, padding, stride, dilation, groups)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias, padding, stride, dilation, groups = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = F.conv2d(grad_output, weight.flip(2).flip(3), padding=(dilation * (weight.size(2) - 1) + 1 - padding[0], dilation * (weight.size(3) - 1) + 1 - padding[1]), stride=stride, dilation=dilation, groups=groups)
        
        if ctx.needs_input_grad[1]:
            grad_weight = F.conv2d(input, grad_output.transpose(0, 1).flip(2).flip(3), padding=(dilation * (grad_output.size(2) - 1) + 1 - padding[0], dilation * (grad_output.size(3) - 1) + 1 - padding[1]), stride=stride, dilation=dilation, groups=groups).transpose(0, 1)
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

class CustomReLUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = F.relu(input)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        # First convolutional layer with ReLU activation and max pooling
        x = CustomReLUFunction.apply(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = CustomReLUFunction.apply(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = CustomReLUFunction.apply(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = CustomReLUFunction.apply(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x