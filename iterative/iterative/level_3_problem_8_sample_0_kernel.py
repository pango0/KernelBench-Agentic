import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class Conv2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(2).flip(3), None, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input.transpose(0, 1).contiguous().view(groups, input.size(1) // groups, -1), grad_output.contiguous().view(groups, -1, 1, 1), None, groups=groups, padding=(weight.size(2)-1)*dilation, dilation=dilation)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None

class BatchNorm2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, running_mean, running_var, weight, bias, eps=1e-5):
        ctx.save_for_backward(input, running_mean, running_var, weight, bias)
        ctx.eps = eps
        mean = input.mean(dim=[0, 2, 3], keepdim=True)
        var = input.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
        invstd = torch.rsqrt(var + eps)
        y = (input - mean) * invstd
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias
        return y, mean, var, invstd

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, grad_mean, grad_var, grad_invstd):
        input, running_mean, running_var, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        batch_size, channels, height, width = input.size()
        mean = input.mean(dim=[0, 2, 3], keepdim=True)
        var = input.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
        invstd = torch.rsqrt(var + eps)

        grad_input = grad_output.clone()
        if weight is not None:
            grad_input *= weight
        grad_input -= grad_output.mean(dim=[0, 2, 3], keepdim=True)
        grad_input /= (var + eps)
        grad_input *= invstd

        if bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))
        else:
            grad_bias = None

        if weight is not None:
            grad_weight = (grad_output * (input - mean)).sum((0, 2, 3))
        else:
            grad_weight = None

        return grad_input, None, None, grad_weight, grad_bias, None

class ReLUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.relu(input, inplace=False)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = Conv2dFunction.apply
        self.bn1 = BatchNorm2dFunction.apply
        self.relu = ReLUFunction.apply
        self.conv2 = Conv2dFunction.apply
        self.bn2 = BatchNorm2dFunction.apply
        self.downsample = nn.Sequential(
            Conv2dFunction.apply,
            BatchNorm2dFunction.apply,
        ) if stride != 1 or in_channels != out_channels * self.expansion else None
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.stride, padding=1, dilation=1, groups=1)
        out = self.bn1(out, running_mean=self.bn1.running_mean, running_var=self.bn1.running_var, weight=self.bn1.weight, bias=self.bn1.bias, eps=1e-5)
        out = self.relu(out)

        out = self.conv2(out, weight=self.conv2.weight, bias=self.conv2.bias, stride=1, padding=1, dilation=1, groups=1)
        out = self.bn2(out, running_mean=self.bn2.running_mean, running_var=self.bn2.running_var, weight=self.bn2.weight, bias=self.bn2.bias, eps=1e-5)

        out += identity
        out = self.relu(out)

        return out