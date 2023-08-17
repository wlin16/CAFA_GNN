### version 1
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class UNetModule(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=9, dropout=0.05):
#         super(UNetModule, self).__init__()
#         self.lstm1 = nn.LSTM(input_size=in_channels, hidden_size=out_channels, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=out_channels, hidden_size=out_channels, batch_first=True)
#         self.norm = nn.LayerNorm(out_channels)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         x = F.relu(x)
#         x, _ = self.lstm2(x)
#         x = self.norm(x)
#         x = self.dropout(x)
#         return x

# class UNet(nn.Module):
#     def __init__(self, input_channels=1280, num_classes=2, num_layers=6, num_filters=64, kernel_size=9, dropout=0.05):
#         super(UNet, self).__init__()
#         self.down_convs = nn.ModuleList([UNetModule(input_channels if i == 0 else num_filters * 2 ** i, 
#                                                     num_filters * 2 ** (i + 1), kernel_size, dropout) 
#                                          for i in range(num_layers)])
#         self.up_convs = nn.ModuleList([UNetModule(num_filters * 2 ** (i + 2), num_filters * 2 ** (i + 1), kernel_size, dropout) 
#                                        for i in range(num_layers - 1, -1, -1)])
#         self.out = nn.Linear(num_filters, num_classes)

#     def forward(self, x):
#         down_outputs = []
#         for down_conv in self.down_convs:
#             x = down_conv(x)
#             down_outputs.append(x)
#             x = x[:, ::2, :] # 下采样

#         for i, up_conv in enumerate(self.up_convs):
#             x = F.interpolate(x, scale_factor=2, mode='nearest') # 上采样
#             # 如果 x 的序列长度大于 down_outputs[-(i+1)] 的序列长度，删除最后一个元素
#             if x.size(1) > down_outputs[-(i+1)].size(1):
#                 x = x[:, :-1, :]
#             x = torch.cat((x, down_outputs[-(i+1)]), dim=2)
#             x = up_conv(x)
        
#         x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1) # 平均池化
#         x = x.view(x.size(0), -1)
#         x = self.out(x)
#         return torch.sigmoid(x)

# # 随机生成一些蛋白质序列嵌入
# torch.manual_seed(0)
# input_data = torch.randn(3, 300, 1280)  # batch_size=3, sequence_length=300, embedding_size=1280

# # 初始化模型
# model = UNet()

# # 前向传播
# output = model(input_data)
# print(output)





###
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_classes):
#         super(UNet, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
        
#         self.encoder1 = ConvBlock(embed_dim, 64)
#         self.pool1 = nn.MaxPool1d(2)
#         self.encoder2 = ConvBlock(64, 128)
#         self.pool2 = nn.MaxPool1d(2)

#         self.bottleneck = ConvBlock(128, 256)

#         self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
#         self.decoder2 = ConvBlock(256, 128)
#         self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
#         self.decoder1 = ConvBlock(128, 64)

#         self.conv_last = nn.Conv1d(64, num_classes, 1)
        
#     def forward(self, x):
#         # Convert string sequences to numeric sequences, then to embeddings
#         x = self.embedding(x)
#         x = x.transpose(1, 2)  # Reshape from [batch_size, sequence_length, embed_dim] to [batch_size, embed_dim, sequence_length]

#         enc1 = self.encoder1(x)
#         enc2 = self.pool1(enc1)
#         enc2 = self.encoder2(enc2)
#         enc3 = self.pool2(enc2)

#         bottleneck = self.bottleneck(enc3)

#         dec2 = self.upconv2(bottleneck)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
        
#         final = self.conv_last(dec1)
#         final = F.avg_pool1d(final, final.size()[2]).squeeze(2)  # Global average pooling
#         out = F.softmax(final, dim=1)

#         return out



# vocab_size = 10000  # Assuming we have 10000 unique tokens in the vocabulary
# embed_dim = 300     # Dimension of the embeddings
# num_classes = 2     # Number of output classes
# batch_size = 3
# sequence_length = 100  # Assuming each sequence in the batch is of length 100

# model = UNet(vocab_size, embed_dim, num_classes)

# # Sample input
# input = torch.randint(0, vocab_size, (batch_size, sequence_length))

# output = model(input)

# print(output.shape)  # Should be torch.Size([3, 2])



import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModule(nn.Module):
    def __init__(self, num_input, num_hidden, dropout=0.05):
        super(LSTMModule, self).__init__()
        self.lstm1 = nn.LSTM(input_size=num_input, hidden_size=num_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=num_hidden, hidden_size=num_hidden, batch_first=True)
        self.norm = nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = F.silu(x)
        x, _ = self.lstm2(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, dropout):
        super(LSTMClassifier, self).__init__()

        self.lstm = LSTMModule(num_input, num_hidden, dropout) # swish function, silu(x)=x∗σ(x),where σ(x) is the logistic sigmoid.
        
        self.fc = nn.Linear(num_hidden, num_output)
    
    def forward(self, x):
        x = self.lstm(x)
        # Since the output of LSTM is (batch_size, seq_len, num_directions * hidden_size),
        # we take the output from the last time step for classification.
        x = x[:, -1, :]
        x = self.fc(x)
        return x

model = LSTMClassifier(1280, 64, 2, 0.05)
data = torch.randn(3, 300, 1280)
output = model(data)
print(output.shape)  # Expected output: torch.Size([3, 2])
