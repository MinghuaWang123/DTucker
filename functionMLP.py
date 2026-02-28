import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义神经网络模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, negative_slope=0.01):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        return x

# 定义反向操作函数
def leakrelu(x,a):
    return np.maximum(a*x, x)

def linear_inverse(y, W, b):
    W_inv = W.T  # 使用伪逆计算
    x = np.dot(y - b, W_inv.T)
    return x


def leaky_relu_derivative(x, alpha=0.01):
    """
    Compute the derivative of the Leaky ReLU function.

    Parameters:
    x : float or np.ndarray
        The input value or array of values.
    alpha : float
        The negative slope coefficient, default is 0.01.

    Returns:
    float or np.ndarray
        The derivative of the Leaky ReLU function.
    """
    # Check if the input is a numpy array or a single float value
    if isinstance(x, (list, np.ndarray)):
        # Convert list to np.ndarray if needed
        x = np.array(x)
        return np.where(x >= 0, 1, alpha)
    else:
        # Handle single float value
        return 1 if x >= 0 else alpha


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU function.

    Parameters:
    x (float or array-like): Input value(s).
    alpha (float): Slope of the function when x < 0. Default is 0.01.

    Returns:
    float or array-like: Output value(s) after applying the Leaky ReLU function.
    """
    return x if x.any() >= 0 else alpha * x

def gradient_descent(C1, C2, W1, b1, W2, b2, beta=1.0, negative_slope=0.01):

    # Forward pass
    b1 = b1.reshape(W1.shape[0], 1)
    b11 = np.repeat(b1, C1.shape[1], axis=1)
    g = W1 @ C1 + b11
    A = leakrelu(g,negative_slope)
    b2 = b2.reshape(W2.shape[0], 1)
    b22= np.repeat(b2, C1.shape[1], axis=1)
    Z = W2 @ A + b22
    diff = Z - C2

    # Compute gradients
    dZ = diff
    dA = W2.T @ dZ
    dg = dA * leaky_relu_derivative(g)
    dX_nn = W1.T @ dg


    # Total gradient
    dC11 = beta * dX_nn



    return dC11

# 设置模型参数
# input_size = 3
# hidden_size = 100  # 隐藏层大小可以自由设置
# output_size = 3
# negative_slope = 0.01
# model = SimpleModel(input_size, hidden_size, output_size, negative_slope)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # 生成示例数据
# A = torch.randn(16900, 3)  # 随机生成输入数据 A
# B = torch.randn(16900, 3)  # 随机生成输出数据 B
# C = torch.randn(16900, 3)  # 随机生成输入的另一个变量 C
#
# # 训练模型
# for epoch in range(100):
#     # 前向传播
#     output = model(A)
#     loss = criterion(output, B)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 10 == 0:
#         print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
#
# # 获取模型参数
# W1 = model.linear1.weight.detach().numpy()
# b1 = model.linear1.bias.detach().numpy()
#
# W2 = model.linear2.weight.detach().numpy()
# b2 = model.linear2.bias.detach().numpy()
#
# # 使用给定矩阵 C 进行计算
# C_np = C.numpy()
#
# # 计算逆操作
# # 输出层的逆操作
# linear_output_C = linear_inverse(C_np, W2, b2)
#
# # Leaky ReLU层的逆操作
# leaky_relu_inv_output_C = leaky_relu_derivative(linear_output_C, model.negative_slope)
#
# # 输入层的逆操作，得到 D
# D = linear_inverse(leaky_relu_inv_output_C, W1, b1)
#
# # 检查逆操作结果
# print("Inverse operation results on C to get D:")
# print(D)
