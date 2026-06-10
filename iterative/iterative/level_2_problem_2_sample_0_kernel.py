import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConvTranspose2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias, stride, padding, output_padding, groups):
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
        grad_input, grad_weight, grad_bias = None, None, None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(2).flip(3), padding=(weight.size(2)-1)*stride[0] - padding[0], groups=groups)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input, grad_output.transpose(0, 1).flip(2).flip(3), padding=(grad_output.size(2)-1)*stride[0] - padding[0], groups=groups)
        
        if ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=[0, 2, 3])
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

class CustomAddBiasFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        output = input + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_bias = torch.sum(grad_output, dim=[0, 2, 3])
        return grad_input, grad_bias

class CustomClampFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, min_val, max_val):
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        output = torch.clamp(input, min=min_val, max=max_val)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone() * ((input >= ctx.min_val) & (input <= ctx.max_val)).float()
        return grad_input, None, None

class CustomScaleFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, scale):
        ctx.save_for_backward(input, scale)
        output = input * scale
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, scale = ctx.saved_tensors
        grad_input = grad_output.clone() * scale
        grad_scale = torch.sum(grad_output * input, dim=[0, 2, 3])
        return grad_input, grad_scale

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = CustomConvTranspose2dFunction.apply(x, self.conv_transpose.weight, self.bias.view(-1, 1, 1), self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.output_padding, self.conv_transpose.groups)
        x = CustomAddBiasFunction.apply(x, self.bias.view(-1, 1, 1))
        x = CustomClampFunction.apply(x, 0.0, 1.0)
        x = CustomScaleFunction.apply(x, self.scaling_factor)
        x = CustomClampFunction.apply(x, 0.0, 1.0)
        x = CustomScaleFunction.apply(x, 1.0 / self.scaling_factor)
        return x

# Example usage:
batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels,)
scaling_factor = 2.0

model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]