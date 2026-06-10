import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class Conv2dFunction(Function):
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
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(dims=[2, 3]), None, stride, padding, dilation, groups)
        grad_weight = torch.nn.functional.conv2d(input.transpose(0, 1), grad_output.contiguous().transpose(0, 1), None, stride, padding, dilation, groups).transpose(0, 1)
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None

class BatchNorm2dFunction(Function):
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
        running_mean.mul_(momentum).add_(mean.squeeze(), alpha=1 - momentum)
        running_var.mul_(momentum).add_(var.squeeze(), alpha=1 - momentum)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, running_mean, running_var, weight, bias, eps = ctx.saved_tensors
        momentum = 0.9  # Assuming momentum is 0.9 for simplicity
        mean = input.mean(dim=(0, 2, 3), keepdim=True)
        var = input.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        invstd = torch.rsqrt(var + eps)
        dy = grad_output
        dbias = dy.sum(dim=(0, 2, 3)) if bias is not None else None
        dweight = (dy * (input - mean) * invstd).sum(dim=(0, 2, 3)) if weight is not None else None
        dxhat = dy * invstd
        dsigma = -(dxhat * (input - mean)).sum(dim=(0, 2, 3)) / ((var + eps) ** 1.5)
        dmu = -dxhat.sum(dim=(0, 2, 3)) / (var + eps) - 2 * dsigma * (input - mean).sum(dim=(0, 2, 3)) / (var + eps)
        dx = dxhat + dsigma * 2 * (input - mean) / (var + eps) + dmu / input.size(0)
        return dx, None, None, dweight, dbias, None

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

class BasicBlockCustom(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockCustom, self).__init__()
        self.conv1 = Conv2dFunction.apply
        self.bn1 = BatchNorm2dFunction.apply
        self.relu = ReLUFunction.apply
        self.conv2 = Conv2dFunction.apply
        self.bn2 = BatchNorm2dFunction.apply
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.stride, padding=1, dilation=1, groups=1)
        out = self.bn1(out, running_mean=self.bn1.running_mean, running_var=self.bn1.running_var, weight=self.bn1.weight, bias=self.bn1.bias, eps=1e-5)
        out = self.relu(out)

        out = self.conv2(out, weight=self.conv2.weight, bias=self.conv2.bias, stride=1, padding=1, dilation=1, groups=1)
        out = self.bn2(out, running_mean=self.bn2.running_mean, running_var=self.bn2.running_var, weight=self.bn2.weight, bias=self.bn2.bias, eps=1e-5)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = Conv2dFunction.apply
        self.bn1 = BatchNorm2dFunction.apply
        self.relu = ReLUFunction.apply
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockCustom, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockCustom, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockCustom, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockCustom, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockCustom.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                Conv2dFunction.apply,
                BatchNorm2dFunction.apply,
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=2, padding=3, dilation=1, groups=1)
        x = self.bn1(x, running_mean=self.bn1.running_mean, running_var=self.bn1.running_var, weight=self.bn1.weight, bias=self.bn1.bias, eps=1e-5)
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