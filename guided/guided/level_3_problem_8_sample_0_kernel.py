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
            grad_weight = torch.nn.functional.conv2d(input.transpose(0, 1).contiguous().view(groups, input.size(1) // groups, -1), grad_output.contiguous().view(groups, grad_output.size(1), -1).transpose(0, 1), None, (1, 1), (0, 0), (1, 1), 1)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None

class BatchNorm2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, running_mean, running_var, weight, bias, eps=1e-5, momentum=0.1):
        ctx.save_for_backward(input, running_mean, running_var, weight, bias)
        ctx.eps = eps
        ctx.momentum = momentum
        output = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training=False, momentum=momentum, eps=eps)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, running_mean, running_var, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        momentum = ctx.momentum
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            invstd = 1 / torch.sqrt(running_var + eps)
            grad_input = (grad_output * weight * invstd).sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[3]:
            grad_weight = (grad_output * input).sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, None, None, grad_weight, grad_bias, None, None

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
        ) if stride != 1 else None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.stride, padding=1, dilation=1, groups=1)
        out = self.bn1(out, running_mean=self.bn1.running_mean, running_var=self.bn1.running_var, weight=self.bn1.weight, bias=self.bn1.bias, eps=1e-5, momentum=0.1)
        out = self.relu(out)

        out = self.conv2(out, weight=self.conv2.weight, bias=self.conv2.bias, stride=1, padding=1, dilation=1, groups=1)
        out = self.bn2(out, running_mean=self.bn2.running_mean, running_var=self.bn2.running_var, weight=self.bn2.weight, bias=self.bn2.bias, eps=1e-5, momentum=0.1)

        if self.downsample is not None:
            identity = self.downsample(identity, weight=self.downsample[0].weight, bias=self.downsample[0].bias, stride=self.stride, padding=0, dilation=1, groups=1)

        out += identity
        out = self.relu(out)

        return out