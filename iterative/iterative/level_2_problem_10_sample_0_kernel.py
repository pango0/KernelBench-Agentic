import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConvTranspose2dFunction(Function):
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
            grad_input = torch.nn.functional.conv2d(grad_output, weight.flip(dims=[2, 3]), padding=padding, stride=stride, groups=groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.functional.conv2d(input, grad_output.flip(dims=[2, 3]), padding=output_padding, stride=stride, groups=groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None

class CustomMaxPool2dFunction(Function):
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
        grad_input = torch.zeros_like(input, dtype=torch.float32, device=input.device)
        indices = torch.argmax(input.view(*input.shape[:2], -1), dim=-1, keepdim=True)
        grad_input.scatter_(2, indices, grad_output.view(*grad_output.shape[:2], -1))
        return grad_input, None, None, None, None, None

class CustomHardtanhFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, min_val, max_val):
        ctx.min_val = min_val
        ctx.max_val = max_val
        output = torch.clamp(input, min=min_val, max=max_val)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        mask = (output >= ctx.min_val) & (output <= ctx.max_val)
        grad_input = grad_output.clone()
        grad_input[~mask] = 0
        return grad_input, None, None

class CustomMeanFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, dim, keepdim):
        ctx.dim = dim
        ctx.keepdim = keepdim
        output = torch.mean(input, dim=dim, keepdim=keepdim)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim
        grad_input = grad_output * input.numel() / input.size(dim)
        grad_input = grad_input.unsqueeze(dim).expand_as(input)
        if not keepdim:
            grad_input = grad_input.squeeze(dim)
        return grad_input, None, None

class CustomTanhFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        output = torch.tanh(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = grad_output * (1 - output ** 2)
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = CustomConvTranspose2dFunction.apply
        self.maxpool = CustomMaxPool2dFunction.apply
        self.hardtanh = CustomHardtanhFunction.apply
        self.mean = CustomMeanFunction.apply
        self.tanh = CustomTanhFunction.apply

    def forward(self, x):
        x = self.conv_transpose(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        x = self.maxpool(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride, padding=self.maxpool_padding, dilation=self.dilation, ceil_mode=self.ceil_mode)
        x = self.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
        x = self.mean(x, dim=(2, 3), keepdim=True)
        x = self.tanh(x)
        return x

# Initialize weights and biases for the conv transpose layer
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max)
model_new.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, dtype=torch.float32, device='cuda'))
model_new.bias = nn.Parameter(torch.randn(out_channels, dtype=torch.float32, device='cuda'))

# Get inputs
inputs = get_inputs()

# Forward pass
output = model_new(inputs[0].to('cuda'))
print(output.shape)