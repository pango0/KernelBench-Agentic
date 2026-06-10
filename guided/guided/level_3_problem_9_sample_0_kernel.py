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
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = F.conv2d(grad_output, weight.flip(dims=[2, 3]), None, stride=(ctx.stride[0], ctx.stride[1]), padding=(ctx.padding[0], ctx.padding[1]), dilation=(ctx.dilation[0], ctx.dilation[1]), groups=ctx.groups)
        if ctx.needs_input_grad[1]:
            grad_weight = F.conv2d(input.transpose(0, 1).contiguous().view(input.size(1) // ctx.groups, ctx.groups, input.size(2), input.size(3)), grad_output.contiguous(), None, stride=(ctx.stride[0], ctx.stride[1]), padding=(ctx.padding[0], ctx.padding[1]), dilation=(ctx.dilation[0], ctx.dilation[1]), groups=ctx.groups).transpose(0, 1).contiguous()
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None

class BatchNorm2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, running_mean, running_var, weight, bias, eps):
        ctx.save_for_backward(input, running_mean, running_var, weight, bias)
        ctx.eps = eps
        output = F.batch_norm(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=eps)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, running_mean, running_var, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            var_inv = 1 / torch.sqrt(running_var + eps)
            grad_input = weight * var_inv * (grad_output - (running_mean + running_var * (input - running_mean) * var_inv) * var_inv)
        if ctx.needs_input_grad[3]:
            grad_weight = (grad_output * (input - running_mean)).sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, None, None, grad_weight, grad_bias, None

class ReLUFunction(Function):
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

class BasicBlockCustom(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockCustom, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = ReLUFunction.apply
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = Conv2dFunction.apply(x, self.conv1.weight, self.conv1.bias, self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups)
        out = BatchNorm2dFunction.apply(out, self.bn1.running_mean, self.bn1.running_var, self.bn1.weight, self.bn1.bias, self.bn1.eps)
        out = self.relu(out)

        out = Conv2dFunction.apply(out, self.conv2.weight, self.conv2.bias, self.conv2.stride, self.conv2.padding, self.conv2.dilation, self.conv2.groups)
        out = BatchNorm2dFunction.apply(out, self.bn2.running_mean, self.bn2.running_var, self.bn2.weight, self.bn2.bias, self.bn2.eps)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
        x = Conv2dFunction.apply(x, self.conv1.weight, self.conv1.bias, self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups)
        x = BatchNorm2dFunction.apply(x, self.bn1.running_mean, self.bn1.running_var, self.bn1.weight, self.bn1.bias, self.bn1.eps)
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