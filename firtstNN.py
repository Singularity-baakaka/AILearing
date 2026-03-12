import numpy as np
import matplotlib.pyplot as plt

NEURONS = 100          # 隐藏层神经元数量
INPUT_SIZE = 1         # 输入层大小
HIDDEN_SIZE = NEURONS  # 隐藏层大小
OUTPUT_SIZE = 1        # 输出层大小
EPOCHS = 1000          # 训练轮数
LEARNING_RATE = 0.01  # 学习率


def gaussian_activation(x):
    return np.exp(-x**2)

def generate_training_data():
    x = np.arange(1, 10.1, 0.1)
    y = np.log(x)
    return x, y


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络结构

        参数:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重和偏置
        # 输入层到隐藏层的权重矩阵，维度为 (input_size, hidden_size)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        # 隐藏层的偏置向量，维度为 (hidden_size,)
        self.bias_hidden = np.random.randn(hidden_size)
        # 隐藏层到输出层的权重矩阵，维度为 (hidden_size, output_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        # 输出层的偏置向量，维度为 (output_size,)
        self.bias_output = np.random.randn(output_size)

    def forward(self, x):
        """
        前向传播过程

        参数:
            x: 输入数据，维度为 (batch_size, input_size)

        返回:
            output: 网络输出，维度为 (batch_size, output_size)
        """
        # 隐藏层输入 = 输入 × 权重 + 偏置
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        # 隐藏层输出 = 激活函数(隐藏层输入)
        self.hidden_output = gaussian_activation(self.hidden_input)
        # 输出层 = 隐藏层输出 × 权重 + 偏置
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        return self.output

    def backward(self, x, y_true, learning_rate):
        """
        反向传播过程，更新网络参数

        参数:
            x: 输入数据，维度为 (batch_size, input_size)
            y_true: 真实标签，维度为 (batch_size, output_size)
            learning_rate: 学习率

        返回:
            loss: 当前样本的损失值
        """
        # 计算输出层误差
        error = self.output - y_true

        # 计算隐藏层到输出层的梯度
        # 权重梯度 = 隐藏层输出 × 误差
        d_weights_hidden_output = np.outer(self.hidden_output, error)
        # 偏置梯度 = 误差
        d_bias_output = error

        # 计算隐藏层误差
        # 隐藏层误差 = 误差 × 权重转置 × 激活函数导数
        # 高斯激活函数的导数: d/dx exp(-x²) = -2x * exp(-x²) = -2x * 激活值
        hidden_error = np.dot(error, self.weights_hidden_output.T) * (-2 * self.hidden_input) * self.hidden_output

        # 计算输入层到隐藏层的梯度
        # 权重梯度 = 输入 × 隐藏层误差
        d_weights_input_hidden = np.outer(x, hidden_error)
        # 偏置梯度 = 隐藏层误差
        d_bias_hidden = hidden_error

        # 使用梯度下降更新参数
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_output -= learning_rate * d_bias_output
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_hidden -= learning_rate * d_bias_hidden

        # 计算均方误差损失
        loss = np.mean(error**2)
        return loss

    def train(self, x_data, y_data, epochs, learning_rate):
        """
        训练神经网络

        参数:
            x_data: 训练数据输入，维度为 (n_samples,)
            y_data: 训练数据标签，维度为 (n_samples,)
            epochs: 训练轮数
            learning_rate: 学习率

        返回:
            losses: 每轮训练的平均损失列表
        """
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            # 遍历所有训练样本
            for x, y in zip(x_data, y_data):
                # 将数据转换为合适的维度
                x = np.array([x])  # 形状: (1, 1)
                y = np.array([y])  # 形状: (1, 1)

                # 前向传播
                self.forward(x)
                # 反向传播并更新参数
                loss = self.backward(x, y, learning_rate)
                total_loss += loss

            # 计算本轮平均损失
            avg_loss = total_loss / len(x_data)
            losses.append(avg_loss)

            # 每100轮打印一次训练进度
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss}")

        return losses

def train_model():
    """
    训练神经网络模型并返回训练结果
    """
    # 生成训练数据
    x_data, y_data = generate_training_data()

    # 初始化神经网络
    model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # 训练模型
    losses = model.train(x_data, y_data, EPOCHS, LEARNING_RATE)

    return x_data, y_data, model, losses



# 交互层
def plot_results(x_data, y_data, model, losses):
    """
    使用matplotlib库绘制训练过程中的损失值变化,拟合函数和真实函数的图形同时显示
    """
    # 创建图形窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(range(len(losses)), losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    # 绘制拟合结果对比
    x_range = np.linspace(min(x_data), max(x_data), 100)
    y_pred = []
    for x in x_range:
        y_pred.append(model.forward(np.array([x]))[0])

    ax2.plot(x_data, y_data, 'b.', label='True data', alpha=0.6)
    ax2.plot(x_range, y_pred, 'r-', label='Fitted curve', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Function Fitting')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 训练模型
x_data, y_data, model, losses = train_model()

# 绘制结果
plot_results(x_data, y_data, model, losses)
print("训练完成")  # 训练完成