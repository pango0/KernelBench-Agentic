import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConvTranspose3dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0)):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        output = torch.nn.functional.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups=1)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv3d(grad_output, weight.flip(2).flip(3).flip(4), padding=ctx.padding, stride=ctx.stride)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv_transpose3d(input, grad_output, padding=ctx.padding, stride=ctx.stride, output_padding=ctx.output_padding)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=[0, 2, 3, 4])
        return grad_input, grad_weight, grad_bias, None, None, None

class CustomLayerNormFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, normalized_shape, eps=1e-5, weight=None, bias=None):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.weight = weight
        ctx.bias = bias
        mean = input.mean(dim=-1, keepdim=True)
        var = ((input - mean) ** 2).mean(dim=-1, keepdim=True)
        inv_var = (var + eps).rsqrt()
        y = (input - mean) * inv_var
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(y, inv_var, mean, var)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        y, inv_var, mean, var = ctx.saved_tensors
        batch_size, channels, depth, height, width = grad_output.size()
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            dy = grad_output
            dvar = -0.5 * (inv_var ** 3) * torch.sum(dy * (y - mean), dim=(-1,), keepdim=True)
            dmean = -torch.sum(dy * inv_var, dim=(-1,), keepdim=True) - 2 * dvar * torch.sum((y - mean), dim=(-1,), keepdim=True) / depth / height / width
            grad_input = dy * inv_var + dvar * 2 * (y - mean) / depth / height / width + dmean / depth / height / width
        if ctx.needs_input_grad[3]:
            grad_weight = torch.sum(grad_output * y, dim=(0, 2, 3, 4))
        if ctx.needs_input_grad[4]:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3, 4))
        return grad_input, None, None, grad_weight, grad_bias

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = CustomConvTranspose3dFunction.apply
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = CustomLayerNormFunction.apply
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x, weight=torch.randn(out_channels, in_channels, *kernel_size).cuda())
        x = x + self.sum_weight
        x = self.norm(x, normalized_shape=norm_shape)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x