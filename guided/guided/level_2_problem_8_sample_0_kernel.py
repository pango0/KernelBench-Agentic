import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class Conv3dCustom(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.nn.functional.conv3d(input, weight, bias, stride=1, padding=1, dilation=1, groups=1)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv_transpose3d(grad_output, weight, bias=None, stride=1, padding=1, output_padding=0, groups=1, dilation=1)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv3d(input, grad_output, bias=None, stride=1, padding=1, dilation=1, groups=1)
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3, 4))
        
        return grad_input, grad_weight, grad_bias

class MaxPool3dCustom(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        output = torch.nn.functional.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
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
        grad_input = torch.nn.functional.max_unpool3d(grad_output, input.argmax(dim=1, keepdim=True), kernel_size, stride, padding, dilation, ceil_mode)
        return grad_input, None, None, None, None, None

class GlobalAvgPool3dCustom(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.mean(input, dim=(2, 3, 4), keepdim=True)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (input.size(2) * input.size(3) * input.size(4)) / input.numel()
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = Conv3dCustom.apply
        self.divisor = divisor
        self.max_pool = MaxPool3dCustom.apply
        self.global_avg_pool = GlobalAvgPool3dCustom.apply
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x, torch.randn(out_channels, in_channels, *kernel_size))
        x = x / self.divisor
        x = self.max_pool(x, pool_size)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x