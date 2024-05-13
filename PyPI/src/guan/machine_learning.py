# Module: machine_learning

# 全连接神经网络模型（包含一个隐藏层）（模型的类定义成全局的）
def fully_connected_neural_network_with_one_hidden_layer(input_size=1, hidden_size=10, output_size=1, activation='relu'):
    import torch
    global model_class_of_fully_connected_neural_network_with_one_hidden_layer
    class model_class_of_fully_connected_neural_network_with_one_hidden_layer(torch.nn.Module):
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
            elif activation == 'gelu':
                hidden_output = torch.nn.functional.gelu(self.hidden_layer(x))
            elif activation == 'silu':
                hidden_output = torch.nn.functional.silu(self.hidden_layer(x))
            else:
                hidden_output = self.hidden_layer(x)
            output = self.output_layer(hidden_output)
            return output
    model = model_class_of_fully_connected_neural_network_with_one_hidden_layer()
    return model

# 全连接神经网络模型（包含两个隐藏层）（模型的类定义成全局的）
def fully_connected_neural_network_with_two_hidden_layers(input_size=1, hidden_size_1=10, hidden_size_2=10, output_size=1, activation_1='relu', activation_2='relu'):
    import torch
    global model_class_of_fully_connected_neural_network_with_two_hidden_layers
    class model_class_of_fully_connected_neural_network_with_two_hidden_layers(torch.nn.Module):
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
            elif activation_1 == 'gelu':
                hidden_output_1 = torch.nn.functional.gelu(self.hidden_layer_1(x))
            elif activation_1 == 'silu':
                hidden_output_1 = torch.nn.functional.silu(self.hidden_layer_1(x))
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
            elif activation_2 == 'gelu':
                hidden_output_2 = torch.nn.functional.gelu(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'silu':
                hidden_output_2 = torch.nn.functional.silu(self.hidden_layer_2(hidden_output_1))
            else:
                hidden_output_2 = self.hidden_layer_2(hidden_output_1)
            
            output = self.output_layer(hidden_output_2)
            return output
    model = model_class_of_fully_connected_neural_network_with_two_hidden_layers()
    return model

# 全连接神经网络模型（包含三个隐藏层）（模型的类定义成全局的）
def fully_connected_neural_network_with_three_hidden_layers(input_size=1, hidden_size_1=10, hidden_size_2=10, hidden_size_3=10, output_size=1, activation_1='relu', activation_2='relu', activation_3='relu'):
    import torch
    global model_class_of_fully_connected_neural_network_with_three_hidden_layers
    class model_class_of_fully_connected_neural_network_with_three_hidden_layers(torch.nn.Module):
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
            elif activation_1 == 'gelu':
                hidden_output_1 = torch.nn.functional.gelu(self.hidden_layer_1(x))
            elif activation_1 == 'silu':
                hidden_output_1 = torch.nn.functional.silu(self.hidden_layer_1(x))
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
            elif activation_2 == 'gelu':
                hidden_output_2 = torch.nn.functional.gelu(self.hidden_layer_2(hidden_output_1))
            elif activation_2 == 'silu':
                hidden_output_2 = torch.nn.functional.silu(self.hidden_layer_2(hidden_output_1))
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
            elif activation_3 == 'gelu':
                hidden_output_3 = torch.nn.functional.gelu(self.hidden_layer_3(hidden_output_2))
            elif activation_3 == 'silu':
                hidden_output_3 = torch.nn.functional.silu(self.hidden_layer_3(hidden_output_2))
            else:
                hidden_output_3 = self.hidden_layer_3(hidden_output_2)
            
            output = self.output_layer(hidden_output_3)
            return output
    model = model_class_of_fully_connected_neural_network_with_three_hidden_layers()
    return model

# 卷积神经网络模型（包含两个卷积层和两个全连接层）（模型的类定义成全局的）
def convolutional_neural_network_with_two_convolutional_layers_and_two_fully_connected_layers(in_channels=1, out_channels_1=10, out_channels_2=10, kernel_size_1=3, kernel_size_2=3, stride_1=1, stride_2=1, padding_1=0, padding_2=0, pooling=1, pooling_kernel_size=2, pooling_stride=2, input_size=1, hidden_size_1=10, hidden_size_2=10, output_size=1):
    import torch
    global model_class_of_convolutional_neural_network_with_two_convolutional_layers_and_two_fully_connected_layers
    class model_class_of_convolutional_neural_network_with_two_convolutional_layers_and_two_fully_connected_layers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convolutional_layer_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=kernel_size_1, stride=stride_1, padding=padding_1)
            self.convolutional_layer_2 = torch.nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size_2, stride=stride_2, padding=padding_2)
            self.pooling_layer = torch.nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride)
            self.hidden_layer_1 = torch.nn.Linear(input_size, hidden_size_1)
            self.hidden_layer_2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
            self.output_layer = torch.nn.Linear(hidden_size_2, output_size)
        def forward(self, x):
            if pooling == 1:
                channel_output_1 = torch.nn.functional.relu(self.pooling_layer(self.convolutional_layer_1(x))) 
                channel_output_2 = torch.nn.functional.relu(self.pooling_layer(self.convolutional_layer_2(channel_output_1)))
            else:
                channel_output_1 = torch.nn.functional.relu(self.convolutional_layer_1(x)) 
                channel_output_2 = torch.nn.functional.relu(self.convolutional_layer_2(channel_output_1))
            channel_output_2 = torch.flatten(channel_output_2, 1)
            hidden_output_1 = torch.nn.functional.relu(self.hidden_layer_1(channel_output_2))
            hidden_output_2 = torch.nn.functional.relu(self.hidden_layer_2(hidden_output_1))
            output = self.output_layer(hidden_output_2)
            return output
    model = model_class_of_convolutional_neural_network_with_two_convolutional_layers_and_two_fully_connected_layers()
    return model

# 使用优化器训练模型
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
                print(epoch, loss.item())
    return model, losses

# 使用优化器批量训练模型
def batch_train_model(model, train_loader, optimizer='Adam', learning_rate=0.001, criterion='MSELoss', num_epochs=1000, print_show=1):
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
        for batch_x, batch_y in train_loader:
            output = model.forward(batch_x)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if print_show == 1:
            if (epoch + 1) % 100 == 0:
                print(epoch, loss.item())
    return model, losses

# 保存模型参数到文件
def save_model_parameters(model, filename='./model_parameters.pth'):
    import torch
    torch.save(model.state_dict(), filename)

# 保存完整模型到文件（保存时需要模型的类可访问）
def save_model(model, filename='./model.pth'):
    import torch
    torch.save(model, filename)

# 加载模型参数（需要输入模型，加载后，原输入的模型参数也会改变）
def load_model_parameters(model, filename='./model_parameters.pth'):
    import torch
    model.load_state_dict(torch.load(filename))
    return model

# 加载完整模型（不需要输入模型，但加载时需要原定义的模型的类可访问）
def load_model(filename='./model.pth'):
    import torch
    model = torch.load(filename)
    return model

# 加载训练数据，用于批量加载训练
def load_train_data(x_train, y_train, batch_size=32):
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# 从pickle文件（多个小文件）中读取输入数据和输出数据，用于训练或预测
def load_input_data_and_output_data_as_torch_tensors_with_pickle(index_range=[1, 2, 3], directory='./', input_filename='input_index=', output_filename='output_index=', type=None):
    import guan
    import numpy as np
    import torch
    input_data = []
    for index in index_range:
        input = guan.load_data(filename=directory+input_filename+str(index))
        input_data.append(input)
    output_data = []
    for index in index_range:
        output = guan.load_data(filename=directory+output_filename+str(index))
        output_data.append(output)
    if type == None:
        input_data = np.array(input_data)
        output_data= np.array(output_data)
    else:
        input_data = np.array(input_data).astype(type)
        output_data= np.array(output_data).astype(type)
    input_data = torch.from_numpy(input_data)
    output_data = torch.from_numpy(output_data)
    return input_data, output_data

# 从pickle文件（一个数组文件）中读取输入数据和输出数据，用于训练或预测
def load_input_data_and_output_data_as_torch_tensors_with_pickle_from_array_file(input_filename='input_file', output_filename='output_file', type=None):
    import guan
    import numpy as np
    import torch
    input_data = guan.load_data(filename=input_filename)
    output_data = guan.load_data(filename=output_filename)
    if type == None:
        input_data = np.array(input_data)
        output_data= np.array(output_data)
    else:
        input_data = np.array(input_data).astype(type)
        output_data= np.array(output_data).astype(type)
    input_data = torch.from_numpy(input_data)
    output_data = torch.from_numpy(output_data)
    return input_data, output_data

# 数据的主成分分析PCA
def pca_of_data(data, n_components=None, standard=1):
    from sklearn.decomposition import PCA
    if standard==1:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
    if n_components==None:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    data_transformed = pca.fit_transform(data_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    return data_transformed, explained_variance_ratio