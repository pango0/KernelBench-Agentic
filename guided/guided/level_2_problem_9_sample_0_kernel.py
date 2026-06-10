import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class LinearFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(input.t(), grad_output)
        grad_bias = torch.sum(grad_output, dim=0) if bias is not None else None
        return grad_input, grad_weight, grad_bias

class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA operators for linear and ReLU operations.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = LinearFunction.apply
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x, self.weight, self.bias)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = torch.relu(x)
        return x

# Initialize weights and bias for the linear layer
model_new = ModelNew(in_features, out_features, subtract_value, multiply_value)
model_new.weight = nn.Parameter(torch.randn(out_features, in_features))
model_new.bias = nn.Parameter(torch.randn(out_features))

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]