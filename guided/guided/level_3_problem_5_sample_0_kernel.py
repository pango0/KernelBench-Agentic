import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class Conv2dFunction(Function):
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
            grad_weight = torch.nn.functional.conv2d(input, grad_output.transpose(0, 1).flip(2).flip(3), None, stride, padding, dilation, groups).transpose(0, 1)
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

class ReLUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.relu(input)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input

class MaxPool2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.save_for_backward(input)
        output = torch.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        indices = torch.argmax(input.view(input.size(0), input.size(1), -1), dim=-1)
        indices = indices.unsqueeze(-1).expand_as(input)
        grad_input.scatter_(2, indices, grad_output)
        return grad_input

class FlattenFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_shape = input.shape
        output = input.view(input.size(0), -1)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.view(ctx.save_shape)

class LinearFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.mm(input, weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), input)
        grad_bias = torch.sum(grad_output, dim=0)
        return grad_input, grad_weight, grad_bias

class DropoutFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, p=0.5, training=True):
        ctx.p = p
        ctx.training = training
        if training:
            mask = torch.rand_like(input) > p
            ctx.mask = mask.to(torch.float32) / (1 - p)
            output = input * ctx.mask
        else:
            output = input
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        if ctx.training:
            return grad_output * ctx.mask, None, None
        else:
            return grad_output, None, None

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
    
    def forward(self, x):
        x = Conv2dFunction.apply(x, self.conv1.weight, self.conv1.bias, self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups)
        x = ReLUFunction.apply(x)
        x = MaxPool2dFunction.apply(x, self.maxpool1.kernel_size, self.maxpool1.stride, self.maxpool1.padding, self.maxpool1.dilation, self.maxpool1.ceil_mode)
        
        x = Conv2dFunction.apply(x, self.conv2.weight, self.conv2.bias, self.conv2.stride, self.conv2.padding, self.conv2.dilation, self.conv2.groups)
        x = ReLUFunction.apply(x)
        x = MaxPool2dFunction.apply(x, self.maxpool2.kernel_size, self.maxpool2.stride, self.maxpool2.padding, self.maxpool2.dilation, self.maxpool2.ceil_mode)
        
        x = Conv2dFunction.apply(x, self.conv3.weight, self.conv3.bias, self.conv3.stride, self.conv3.padding, self.conv3.dilation, self.conv3.groups)
        x = ReLUFunction.apply(x)
        
        x = Conv2dFunction.apply(x, self.conv4.weight, self.conv4.bias, self.conv4.stride, self.conv4.padding, self.conv4.dilation, self.conv4.groups)
        x = ReLUFunction.apply(x)
        
        x = Conv2dFunction.apply(x, self.conv5.weight, self.conv5.bias, self.conv5.stride, self.conv5.padding, self.conv5.dilation, self.conv5.groups)
        x = ReLUFunction.apply(x)
        x = MaxPool2dFunction.apply(x, self.maxpool3.kernel_size, self.maxpool3.stride, self.maxpool3.padding, self.maxpool3.dilation, self.maxpool3.ceil_mode)
        
        x = FlattenFunction.apply(x)
        
        x = LinearFunction.apply(x, self.fc1.weight, self.fc1.bias)
        x = ReLUFunction.apply(x)
        x = DropoutFunction.apply(x, self.dropout1.p, self.dropout1.training)
        
        x = LinearFunction.apply(x, self.fc2.weight, self.fc2.bias)
        x = ReLUFunction.apply(x)
        x = DropoutFunction.apply(x, self.dropout2.p, self.dropout2.training)
        
        x = LinearFunction.apply(x, self.fc3.weight, self.fc3.bias)
        
        return x