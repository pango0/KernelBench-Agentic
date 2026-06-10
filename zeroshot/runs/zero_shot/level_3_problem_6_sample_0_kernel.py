import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomConv2dFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias, stride, padding, dilation, groups)
        output = torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias, stride, padding, dilation, groups = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.conv2d(grad_output, weight.flip(2).flip(3), None, stride, padding, dilation, groups)
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.conv2d(input.transpose(0, 1).contiguous().view(input.size(1), -1, input.size(2), input.size(3)), 
                                      grad_output.transpose(0, 1).contiguous().view(grad_output.size(1), -1, grad_output.size(2), grad_output.size(3)), 
                                      None, groups=groups, padding=(weight.size(2)-1)*dilation, stride=stride)
        
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None, None, None

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CustomConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    def forward(self, input):
        return CustomConv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = CustomConv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            CustomConv2d(in_channels, reduce_3x3, kernel_size=1),
            CustomConv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            CustomConv2d(in_channels, reduce_5x5, kernel_size=1),
            CustomConv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            CustomConv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)