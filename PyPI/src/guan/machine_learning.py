# Module: machine_learning
import guan

# 全连接神经网络模型（包含一个隐藏层）
@guan.function_decorator
def fully_connected_neural_network_with_one_hidden_layer(input_size=1, hidden_size=10, output_size=1, activation='relu'):
    import torch
    class model_class(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_layer = torch.nn.Linear(input_size, hidden_size)
            self.output_layer = torch.nn.Linear(hidden_size, output_size)
        def forward(self, x):
            if activation == 'relu':
                hidden_output = torch.nn.functional.relu(self.hidden_layer(x))
            elif activation == 'leaky_relu':
                hidden_output = torch.nn.functional.leaky_relu(self.hidden_layer(x))
            elif activation == 'sigmoid':
                hidden_output = torch.nn.functional.sigmoid(self.hidden_layer(x))
            elif activation == 'tanh':
                hidden_output = torch.nn.functional.tanh(self.hidden_layer(x))
            else:
                hidden_output = self.hidden_layer(x)
            output = self.output_layer(hidden_output)
            return output
    model = model_class()
    return model

# 全连接神经网络模型（包含两个隐藏层）
@guan.function_decorator
def fully_connected_neural_network_with_two_hidden_layers(input_size=1, hidden_size_1=10, hidden_size_2=10, output_size=1, activation_1='relu', activation_2='relu'):
    import torch
    class model_class(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_layer_1 = torch.nn.Linear(input_size, hidden_size_1)
            self.hidden_layer_2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
            self.output_layer = torch.nn.Linear(hidden_size_2, output_size)
        def forward(self, x):
            if activation_1 == 'relu':
                hidden_output_1 = torch.nn.functional.relu(self.hidden_layer_1(x))
            elif activation_1 == 'leaky_relu':
                hidden_output_1 = torch.nn.functional.leaky_relu(self.hidden_layer_1(x))
            elif activation_1 == 'sigmoid':
                hidden_output_1 = torch.nn.functional.sigmoid(self.hidden_layer_1(x))
            elif activation_1 == 'tanh':
                hidden_output_1 = torch.nn.functional.tanh(self.hidden_layer_1(x))
            else:
                hidden_output_1 = self.hidden_layer_1(x)
            
            if activation_2 == 'relu':
                hidden_output_2 = torch.nn.functional.relu(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'leaky_relu':
                hidden_output_2 = torch.nn.functional.leaky_relu(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'sigmoid':
                hidden_output_2 = torch.nn.functional.sigmoid(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'tanh':
                hidden_output_2 = torch.nn.functional.tanh(self.hidden_layer_2(hidden_output_1))
            else:
                hidden_output_2 = self.hidden_layer_2(hidden_output_1)
            
            output = self.output_layer(hidden_output_2)
            return output
    model = model_class()
    return model

# 全连接神经网络模型（包含三个隐藏层）
@guan.function_decorator
def fully_connected_neural_network_with_three_hidden_layers(input_size=1, hidden_size_1=10, hidden_size_2=10, hidden_size_3=10, output_size=1, activation_1='relu', activation_2='relu', activation_3='relu'):
    import torch
    class model_class(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_layer_1 = torch.nn.Linear(input_size, hidden_size_1)
            self.hidden_layer_2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
            self.hidden_layer_3 = torch.nn.Linear(hidden_size_2, hidden_size_3)
            self.output_layer = torch.nn.Linear(hidden_size_3, output_size)
        def forward(self, x):
            if activation_1 == 'relu':
                hidden_output_1 = torch.nn.functional.relu(self.hidden_layer_1(x))
            elif activation_1 == 'leaky_relu':
                hidden_output_1 = torch.nn.functional.leaky_relu(self.hidden_layer_1(x))
            elif activation_1 == 'sigmoid':
                hidden_output_1 = torch.nn.functional.sigmoid(self.hidden_layer_1(x))
            elif activation_1 == 'tanh':
                hidden_output_1 = torch.nn.functional.tanh(self.hidden_layer_1(x))
            else:
                hidden_output_1 = self.hidden_layer_1(x)
            
            if activation_2 == 'relu':
                hidden_output_2 = torch.nn.functional.relu(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'leaky_relu':
                hidden_output_2 = torch.nn.functional.leaky_relu(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'sigmoid':
                hidden_output_2 = torch.nn.functional.sigmoid(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'tanh':
                hidden_output_2 = torch.nn.functional.tanh(self.hidden_layer_2(hidden_output_1))
            else:
                hidden_output_2 = self.hidden_layer_2(hidden_output_1)

            if activation_3 == 'relu':
                hidden_output_3 = torch.nn.functional.relu(self.hidden_layer_3(hidden_output_2))
            elif activation_3 == 'leaky_relu':
                hidden_output_3 = torch.nn.functional.leaky_relu(self.hidden_layer_3(hidden_output_2))
            elif activation_3 == 'sigmoid':
                hidden_output_3 = torch.nn.functional.sigmoid(self.hidden_layer_3(hidden_output_2))
            elif activation_3 == 'tanh':
                hidden_output_3 = torch.nn.functional.tanh(self.hidden_layer_3(hidden_output_2))
            else:
                hidden_output_3 = self.hidden_layer_3(hidden_output_2)
            
            output = self.output_layer(hidden_output_3)
            return output
    model = model_class()
    return model

# 使用优化器训练模型
@guan.function_decorator
def train_model(model, x_data, y_data, optimizer='Adam', learning_rate=0.001, criterion='MSELoss', num_epochs=1000, print_show=1):
    import torch
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    if criterion == 'MSELoss':
        criterion = torch.nn.MSELoss()
    elif criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(num_epochs):
        output = model.forward(x_data)
        loss = criterion(output, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if print_show == 1:
            if (epoch + 1) % 100 == 0:
                print(epoch)
    return model, losses

# 保存完整模型到文件
@guan.function_decorator
def save_model(model, filename='./model.pth'):
    import torch
    torch.save(model, filename)

# 保存模型参数到文件
@guan.function_decorator
def save_model_parameters(model, filename='./model_parameters.pth'):
    import torch
    torch.save(model.state_dict(), filename)

# 加载完整模型
@guan.function_decorator
def load_model(filename='./model.pth'):
    import torch
    model = torch.load(filename)
    return model

# 加载模型参数（需要输入模型）
@guan.function_decorator
def load_model_parameters(model, filename='./model_parameters.pth'):
    import torch
    model.load_state_dict(torch.load(filename))
    return model