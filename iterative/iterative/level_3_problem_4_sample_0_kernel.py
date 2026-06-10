import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConv2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias, stride, padding, dilation, groups)
        output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias, stride, padding, dilation, groups = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(2).flip(3), None, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input.transpose(1, 0), grad_output, None, stride, padding, dilation, groups).transpose(1, 0)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

class CustomLinearFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.linear(input, weight, bias)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.linear(grad_output, weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = torch.mm(input.t(), grad_output)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=0)
        
        return grad_input, grad_weight, grad_bias

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
        x = CustomConv2dFunction.apply(x, self.conv1.weight, self.conv1.bias)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = CustomConv2dFunction.apply(x, self.conv2.weight, self.conv2.bias)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = CustomLinearFunction.apply(x, self.fc1.weight, self.fc1.bias)
        x = F.relu(x)
        
        # Second fully connected layer with ReLU activation
        x = CustomLinearFunction.apply(x, self.fc2.weight, self.fc2.bias)
        x = F.relu(x)
        
        # Final fully connected layer
        x = CustomLinearFunction.apply(x, self.fc3.weight, self.fc3.bias)
        
        return x