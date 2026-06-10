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
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.conv3d(grad_output, weight.flip(2).flip(3).flip(4), padding=ctx.padding, stride=ctx.stride)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv3d(input, grad_output, padding=ctx.padding, stride=ctx.stride)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=[0, 2, 3, 4])

        return grad_input, grad_weight, grad_bias, None, None, None

class CustomLayerNormFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, normalized_shape, eps, weight=None, bias=None):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.weight = weight
        ctx.bias = bias
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        inv_var = 1 / (var + eps)
        ctx.save_for_backward(input, mean, inv_var)
        output = (input - mean) * inv_var
        if weight is not None:
            output = output * weight
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, mean, inv_var = ctx.saved_tensors
        batch_size = input.size(0)
        channels = input.size(1)
        spatial_dim = input.size(2) * input.size(3) * input.size(4)
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_mean = torch.sum(grad_output, dim=(-1,), keepdim=True)
            grad_var = torch.sum(grad_output * (input - mean) * (-0.5) * inv_var**3, dim=(-1,), keepdim=True)
            grad_input = grad_output * inv_var + grad_mean * (-1.0 / spatial_dim) + grad_var * (2.0 * (input - mean) / spatial_dim)

        if ctx.weight is not None and ctx.needs_input_grad[3]:
            grad_weight = torch.sum(grad_output * (input - mean) * inv_var, dim=(-1,), keepdim=True)

        if ctx.bias is not None and ctx.needs_input_grad[4]:
            grad_bias = torch.sum(grad_output, dim=(-1,), keepdim=True)

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
        x = self.conv_transpose(x, weight=torch.randn(out_channels, in_channels, *kernel_size, device=x.device, dtype=torch.float32))
        x = x + self.sum_weight
        x = self.norm(x, normalized_shape=self.norm.shape, eps=1e-5, weight=torch.ones(out_channels, device=x.device, dtype=torch.float32), bias=torch.zeros(out_channels, device=x.device, dtype=torch.float32))
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x

# Example usage
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size)
inputs = get_inputs()
outputs = model_new(inputs[0])
print(outputs.shape)