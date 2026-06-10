import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class ConvTranspose2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        output = torch.nn.functional.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(2).flip(3), padding=padding, stride=stride, groups=groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input, grad_output.transpose(0, 1).flip(2).flip(3), padding=output_padding, stride=stride, groups=groups)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=[0, 2, 3])

        return grad_input, grad_weight, grad_bias, None, None, None, None

class MaxPool2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        output = torch.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        ceil_mode = ctx.ceil_mode
        grad_input = torch.zeros_like(input)
        indices = torch.argmax(input.view(*input.shape[:2], -1), dim=-1)
        indices = indices.unsqueeze(-1).unsqueeze(-1).expand_as(input)
        grad_input.scatter_(2, indices, grad_output)
        return grad_input, None, None, None, None, None

class HardtanhFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, min_val, max_val):
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        output = torch.clamp(input, min_val, max_val)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        min_val = ctx.min_val
        max_val = ctx.max_val
        mask = (input >= min_val) & (input <= max_val)
        grad_input = grad_output * mask.to(torch.float32)
        return grad_input, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = ConvTranspose2dFunction.apply
        self.maxpool = MaxPool2dFunction.apply
        self.hardtanh = HardtanhFunction.apply

    def forward(self, x):
        x = self.conv_transpose(x, weight=torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=x.device, dtype=torch.float32))
        x = self.maxpool(x, kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        x = self.hardtanh(x, min_val=-1, max_val=1)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = torch.tanh(x)
        return x