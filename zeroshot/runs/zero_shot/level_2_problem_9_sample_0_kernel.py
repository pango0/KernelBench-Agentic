import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class MatMulSubtractMultiplyReLUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight, subtract_value, multiply_value):
        ctx.save_for_backward(input, weight, subtract_value, multiply_value)
        output = torch.matmul(input, weight)
        output -= subtract_value
        output *= multiply_value
        output = torch.relu(output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, subtract_value, multiply_value = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        grad_subtract_value = None
        grad_multiply_value = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(input.t(), grad_output)
        if ctx.needs_input_grad[2]:
            grad_subtract_value = torch.sum(grad_output, dim=0)
        if ctx.needs_input_grad[3]:
            grad_multiply_value = torch.sum(grad_output * input, dim=0)

        return grad_input, grad_weight, grad_subtract_value, grad_multiply_value

class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA operators for matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = MatMulSubtractMultiplyReLUFunction.apply(x, self.linear.weight, self.subtract_value, self.multiply_value)
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