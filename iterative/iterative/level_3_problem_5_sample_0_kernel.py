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
            grad_weight = torch.nn.functional.conv2d(input.transpose(1, 0), grad_output, None, stride, padding, dilation, groups).transpose(1, 0)
        
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
        indices = torch.argmax(input.view(*input.shape[:2], -1), dim=-1)
        indices = indices.unsqueeze(-1).unsqueeze(-1).expand_as(input)
        grad_input.scatter_(2, indices, grad_output)
        return grad_input

class LinearFunction(Function):
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

class DropoutFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, p=0.5, training=True, inplace=False):
        ctx.p = p
        ctx.training = training
        if ctx.training:
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
        
        self.conv1 = Conv2dFunction.apply
        self.relu1 = ReLUFunction.apply
        self.maxpool1 = MaxPool2dFunction.apply
        
        self.conv2 = Conv2dFunction.apply
        self.relu2 = ReLUFunction.apply
        self.maxpool2 = MaxPool2dFunction.apply
        
        self.conv3 = Conv2dFunction.apply
        self.relu3 = ReLUFunction.apply
        
        self.conv4 = Conv2dFunction.apply
        self.relu4 = ReLUFunction.apply
        
        self.conv5 = Conv2dFunction.apply
        self.relu5 = ReLUFunction.apply
        self.maxpool3 = MaxPool2dFunction.apply
        
        self.fc1 = LinearFunction.apply
        self.relu6 = ReLUFunction.apply
        self.dropout1 = DropoutFunction.apply
        
        self.fc2 = LinearFunction.apply
        self.relu7 = ReLUFunction.apply
        self.dropout2 = DropoutFunction.apply
        
        self.fc3 = LinearFunction.apply
    
    def forward(self, x):
        x = self.conv1(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=4, padding=2)
        x = self.relu1(x)
        x = self.maxpool1(x, kernel_size=3, stride=2)
        
        x = self.conv2(x, weight=self.conv2.weight, bias=self.conv2.bias, padding=2)
        x = self.relu2(x)
        x = self.maxpool2(x, kernel_size=3, stride=2)
        
        x = self.conv3(x, weight=self.conv3.weight, bias=self.conv3.bias, padding=1)
        x = self.relu3(x)
        
        x = self.conv4(x, weight=self.conv4.weight, bias=self.conv4.bias, padding=1)
        x = self.relu4(x)
        
        x = self.conv5(x, weight=self.conv5.weight, bias=self.conv5.bias, padding=1)
        x = self.relu5(x)
        x = self.maxpool3(x, kernel_size=3, stride=2)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x, weight=self.fc1.weight, bias=self.fc1.bias)
        x = self.relu6(x)
        x = self.dropout1(x, p=0.0, training=self.training)
        
        x = self.fc2(x, weight=self.fc2.weight, bias=self.fc2.bias)
        x = self.relu7(x)
        x = self.dropout2(x, p=0.0, training=self.training)
        
        x = self.fc3(x, weight=self.fc3.weight, bias=self.fc3.bias)
        
        return x