import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConv2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
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
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(dims=[2, 3]).transpose(0, 1), None, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input.transpose(0, 1), grad_output, None, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
        if ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None

class CustomBatchNorm2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, running_mean, running_var, weight, bias, eps):
        ctx.save_for_backward(input, running_mean, running_var, weight, bias)
        ctx.eps = eps
        mean = input.mean(dim=(0, 2, 3), keepdim=True)
        var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
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
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * grad_invstd
            grad_input = grad_input * weight
            grad_input = grad_input + grad_mean * (-grad_invstd / batch_size)
            grad_input = grad_input + grad_var * (-2 * (input - running_mean) * grad_invstd**3 / batch_size)
        if ctx.needs_input_grad[3]:
            grad_weight = torch.sum(grad_output * (input - running_mean) * grad_invstd, dim=(0, 2, 3))
        if ctx.needs_input_grad[4]:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        return grad_input, None, None, grad_weight, grad_bias, None

class CustomReLUFunction(Function):
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

class BottleneckCustom(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckCustom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = CustomReLUFunction.apply
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = CustomConv2dFunction.apply(x, self.conv1.weight, self.conv1.bias, 1, 0, 1, 1)
        out = CustomBatchNorm2dFunction.apply(out, self.bn1.running_mean, self.bn1.running_var, self.bn1.weight, self.bn1.bias, 1e-5)
        out = self.relu(out)

        out = CustomConv2dFunction.apply(out, self.conv2.weight, self.conv2.bias, 1, 1, 1, 1)
        out = CustomBatchNorm2dFunction.apply(out, self.bn2.running_mean, self.bn2.running_var, self.bn2.weight, self.bn2.bias, 1e-5)
        out = self.relu(out)

        out = CustomConv2dFunction.apply(out, self.conv3.weight, self.conv3.bias, 1, 0, 1, 1)
        out = CustomBatchNorm2dFunction.apply(out, self.bn3.running_mean, self.bn3.running_var, self.bn3.weight, self.bn3.bias, 1e-5)

        if self.downsample is not None:
            identity = CustomConv2dFunction.apply(identity, self.downsample[0].weight, self.downsample[0].bias, 1, 0, 1, 1)
            identity = CustomBatchNorm2dFunction.apply(identity, self.downsample[1].running_mean, self.downsample[1].running_var, self.downsample[1].weight, self.downsample[1].bias, 1e-5)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = CustomReLUFunction.apply
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckCustom

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = CustomConv2dFunction.apply(x, self.conv1.weight, self.conv1.bias, 1, 0, 1, 1)
        x = CustomBatchNorm2dFunction.apply(x, self.bn1.running_mean, self.bn1.running_var, self.bn1.weight, self.bn1.bias, 1e-5)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x