import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomMatMulSubtract(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, subtract_value):
        ctx.save_for_backward(input, weight, subtract_value)
        output = torch.matmul(input, weight) - subtract_value
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, subtract_value = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight.t())
        grad_weight = torch.matmul(input.t(), grad_output)
        grad_subtract_value = torch.sum(grad_output, dim=0)
        return grad_input, grad_weight, None

class CustomMultiplyReLU(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, multiply_value):
        ctx.save_for_backward(input, multiply_value)
        output = input * multiply_value
        ctx.output = output
        return torch.relu(output)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, multiply_value = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = ctx.output > 0
        grad_input[~mask] = 0
        grad_input *= multiply_value
        return grad_input, None

class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA operators for matmul-subtract and multiply-relu.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = CustomMatMulSubtract.apply(x, self.linear.weight, self.subtract_value)
        x = CustomMultiplyReLU.apply(x, self.multiply_value)
        return x

# Example usage:
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

model_new = ModelNew(in_features, out_features, subtract_value, multiply_value)
inputs = get_inputs()

output = model_new(inputs[0])
print(output.shape)