import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConvTranspose2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias, stride, padding, output_padding):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        output = torch.nn.functional.conv_transpose2d(input, weight, bias, stride, padding, output_padding)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(2).flip(3), padding=(weight.size(2)-1)*ctx.stride[0] + ctx.padding[0], dilation=ctx.stride[0])

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input, grad_output.transpose(0, 1).flip(2).flip(3), padding=(input.size(2)-1)*ctx.stride[0] + ctx.padding[0], dilation=ctx.stride[0])

        if ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=[0, 2, 3])

        return grad_input, grad_weight, grad_bias, None, None, None

class CustomTanhFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.tanh(input)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (1 - torch.tanh(input)**2)
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = CustomConvTranspose2dFunction.apply
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.tanh = CustomTanhFunction.apply
        self.register_buffer('weight', torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        x = self.conv_transpose(x, self.weight, self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        x = x - self.bias
        x = self.tanh(x)
        return x