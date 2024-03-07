from mimetypes import init
from turtle import forward
from unicodedata import bidirectional
import torch
from torch import device, nn
import module_utils
from module_utils import conv_blocks, GELU, Residual, SoftmaxPooling1D, TargetLengthCrop
from einops import rearrange
from attention import MultiHeadAttention
from einops.layers.torch import Rearrange
import numpy as np

TARGET_LENGTH = 500


class cnn_module(nn.Sequential):
    def __init__(self, *config):
        super(cnn_module, self).__init__()
        # self.linear_dim = config[0].linear_dim
        self.cnn_output_dim = config[0].cnn_params.cnn_output_dim
        self.conv1 = nn.Conv2d(
            config[0].cnn_params.cnn_input_dim,
            config[0].cnn_params.cnn_output_dim,
            config[0].cnn_params.cnn_kernel,
            stride=config[0].cnn_params.stride,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(
            config[0].cnn_params.cnn_output_dim,
            config[0].cnn_params.cnn_output_dim,
            kernel_size=3,
            stride=1,
        )
        self.dropout = nn.Dropout2d(p=0.05)
        self.fc1 = nn.Linear(
            config[0].cnn_params.cnn_output_dim * 997 * 190, config[0].linear_dim
        )
        self.fc2 = nn.Linear(config[0].linear_dim, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, self.cnn_output_dim * 997 * 190)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 160,120,641
class transformer_module(nn.Module):
    def __init__(self, *config):
        super(transformer_module, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 192, 50, stride=12),
            nn.ReLU(),
            nn.Dropout1d(p=0.05),
            nn.MaxPool1d(kernel_size=3, stride=2, return_indices=True),
        )

        def transformer_mlp():
            return module_utils.Residual(
                nn.Sequential(
                    nn.LayerNorm(config[0].channels),
                    nn.Linear(config[0].channels, config[0].channels * 2),
                    nn.Dropout(config[0].dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config[0].channels * 2, config[0].channels),
                    nn.Dropout(config[0].dropout_rate),
                )
            )

        transformer = []
        for _ in range(config[0].num_transformer_layers):
            transformer.append(
                nn.Sequential(
                    Rearrange("b c l -> b l c"),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config[0].channels),
                            MultiHeadAttention(**config[0].transfomer_params),
                            nn.Dropout(config[0].dropout_rate),
                        )
                    ),
                    transformer_mlp(),
                )
            )

        self.transformer = nn.Sequential(*transformer)
        self.fc1 = nn.Sequential(
            Rearrange("b l c -> b (l c)"), nn.Linear(639 * 192, config[0].linear_dim)
        )

        self.fc2 = nn.Linear(config[0].linear_dim, 2)

    def forward(self, x):

        x = torch.sum(x, axis=1)

        # x,_ = torch.max(x,dim=0)
        x = torch.unsqueeze(x, 1)
        x, indices = self.conv_block(x)
        # x = self.transformer(x)
        x = self.transformer(x)
        xx = self.fc1(x)
        x = nn.functional.relu(self.fc1(x))

        x = self.fc2(x)
        return x, xx


class cnn2d_transformer_module(nn.Module):
    def __init__(self, *config):
        super(cnn2d_transformer_module, self).__init__()
        # self.linear_dim = config[0].linear_dim
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 192, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(1, 1, 2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        def transformer_mlp():
            return module_utils.Residual(
                nn.Sequential(
                    nn.LayerNorm(config[0].channels),
                    nn.Linear(config[0].channels, config[0].channels * 2),
                    nn.Dropout(config[0].dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config[0].channels * 2, config[0].channels),
                    nn.Dropout(config[0].dropout_rate),
                )
            )

        transformer = []
        for _ in range(config[0].num_transformer_layers):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config[0].channels),
                            MultiHeadAttention(**config[0].transfomer_params),
                            nn.Dropout(config[0].dropout_rate),
                        )
                    ),
                    transformer_mlp(),
                )
            )

        self.transformer = nn.Sequential(*transformer)

        self.fc1 = nn.Sequential(
            Rearrange("b l c -> b (l c)"), nn.Linear(999 * 192, config[0].linear_dim)
        )
        self.fc2 = nn.Linear(config[0].linear_dim, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.squeeze(x, 1)
        x = self.transformer(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class cnn_transformer_module(nn.Module):
    def __init__(self, *config):
        super(cnn_transformer_module, self).__init__()
        self.conv_1 = nn.Conv1d(
            config[0].cnn_params.input_dim,
            config[0].cnn_params.filter_dim,
            32,
            stride=16,
        )
        self.conv_block = nn.Sequential(
            nn.Sequential(
                nn.BatchNorm1d(config[0].cnn_params.channels // 2),
                module_utils.GELU(),
                nn.Conv1d(
                    config[0].cnn_params.channels // 2,
                    config[0].cnn_params.channels // 2,
                    3,
                    padding="same",
                ),
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        filter_list = [384, 768]
        conv_layers = []
        for in_channels, out_channels in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    conv_blocks(in_channels, out_channels, 5),
                    module_utils.Residual(conv_blocks(out_channels, out_channels, 1)),
                    module_utils.SoftmaxPooling1D(out_channels, pool_size=2),
                )
            )
        self.conv_tower = nn.Sequential(*conv_layers)

        def transformer_mlp():
            return module_utils.Residual(
                nn.Sequential(
                    nn.LayerNorm(config[0].cnn_params.channels),
                    nn.Linear(
                        config[0].cnn_params.channels, config[0].cnn_params.channels * 2
                    ),
                    nn.Dropout(config[0].dropout_rate),
                    nn.ReLU(),
                    nn.Linear(
                        config[0].cnn_params.channels * 2, config[0].cnn_params.channels
                    ),
                    nn.Dropout(config[0].dropout_rate),
                )
            )

        transformer = []
        for _ in range(config[0].num_transformer_layers):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config[0].cnn_params.channels),
                            MultiHeadAttention(**config[0].transfomer_params),
                            nn.Dropout(config[0].dropout_rate),
                        )
                    ),
                    transformer_mlp(),
                )
            )

        self.transformer = nn.Sequential(Rearrange("b c l -> b l c"), *transformer)
        self.crop_final = TargetLengthCrop(TARGET_LENGTH)
        self.final_pointwise = nn.Sequential(
            nn.Linear(
                config[0].cnn_params.channels,
                int(config[0].cnn_params.channels * 0.25),
                1,
            ),
            nn.Dropout(config[0].dropout_rate / 4),
            GELU(),
        )
        self._trunk = nn.Sequential(
            self.conv_block, self.conv_tower, self.transformer, self.final_pointwise
        )

        self.head = nn.Sequential(
            Rearrange("b l c -> b (l c)"),
            nn.Linear(int(config[0].cnn_params.channels * 0.25 * 2125), 2),
        )

    def forward(self, x):
        batch_size, rt_len = x.size()[0], x.size()[1]
        x = x.reshape(batch_size * rt_len, x.size()[-1])
        x = torch.unsqueeze(x, 1)
        x = self.conv_1(x)
        x = torch.squeeze(x, 1)
        x = rearrange(x, "(b l) c -> b c l", b=batch_size)
        x = self._trunk(x)
        # x = self.conv_block(x)
        # x = self.conv_tower(x)
        # x = self.transformer(x)
        # #x = self.crop_final(x)
        # x = self.final_pointwise(x)
        x = self.head(x)
        return x


class dilated_cnn_module(nn.Sequential):
    def __init__(self, *config):
        super(dilated_cnn_module, self).__init__()
        self.cnn_output_dim = config[0].cnn_params.cnn_output_dim
        self.conv1 = nn.Conv2d(
            config[0].cnn_params.cnn_input_dim,
            1,
            kernel_size=5,
            stride=2,
            padding=1,
            dilation=1,
        )
        # self.conv2 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(
            1,
            config[0].cnn_params.cnn_output_dim,
            kernel_size=5,
            stride=2,
            padding=4,
            dilation=4,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout2d(p=0.05)
        self.fc1 = nn.Linear(
            config[0].cnn_params.cnn_output_dim * 497 * 93, config[0].linear_dim
        )
        self.fc2 = nn.Linear(config[0].linear_dim, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.dropout(self.conv3(x)), 2))
        x = x.view(-1, self.cnn_output_dim * 497 * 93)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_RNN(nn.Sequential):

    def __init__(self, *config):
        super(CNN_RNN, self).__init__()

        # self.conv = nn.ModuleList([nn.Conv1d(**config.cnn_params) for i in range(config.cnn_layers_num)])
        self.num_layers = config[0].rnn_params.rnn_num_layers
        self.rnn_hidden_dim = config[0].rnn_params.rnn_hidden_dim
        self.linear_dim = config[0].linear_dim
        self.rnn_name = config[0].rnn_name
        self.conv = nn.Conv1d(
            config[0].cnn_params.cnn_input_dim,
            config[0].cnn_params.cnn_output_dim,
            config[0].cnn_params.cnn_kernel,
            stride=config[0].cnn_params.stride,
        )
        # self.conv = self.conv.double()
        self.max_pooling = nn.MaxPool1d(3, stride=2)
        self.fc1 = nn.Linear(615, config[0].linear_dim)
        if config[0].rnn_name == "LSTM":
            self.rnn = nn.LSTM(
                input_size=config[0].linear_dim,
                hidden_size=config[0].rnn_params.rnn_hidden_dim,
                num_layers=config[0].rnn_params.rnn_num_layers,
                bidirectional=config[0].rnn_params.rnn_bidirectional,
                batch_first=True,
            )
        elif config[0].rnn_name == "GRU":
            self.rnn = nn.GRU(
                input_size=config[0].linear_dim,
                hidden_size=config[0].rnn_params.rnn_hidden_dim,
                num_layers=config[0].rnn_params.rnn_num_layers,
                bidirectional=config[0].rnn_params.rnn_bidirectional,
                batch_first=True,
            )
        else:
            raise AttributeError("please use LSTM or GRU")
        self.direction = 2 if config[0].rnn_params.rnn_bidirectional else 1
        # self.rnn = self.rnn.double()
        self.fc2 = nn.Linear(
            config[0].rnn_params.rnn_hidden_dim * self.direction, config[0].label_num
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, rt_len = x.size()[0], x.size()[1]
        x_input = x.reshape(batch_size * rt_len, x.size()[-1])
        x_input1 = torch.unsqueeze(x_input, 1)
        x_input2 = self.max_pooling(nn.functional.relu(self.conv(x_input1)))
        output_ = torch.squeeze(x_input2, 1)
        # output_ = output.reshape(batch_size,rt_len,1439)
        output_temp = self.fc1(output_)
        cnn_output = output_temp.reshape(batch_size, rt_len, self.linear_dim)
        # cnn_output = cnn_output.view(cnn_output.size(0), cnn_output.size(1), -1)
        # import pdb; pdb.set_trace()
        if self.rnn_name == "LSTM":
            # h0 = torch.randn(self.num_layers*self.direction,batch_size,self.rnn_hidden_dim).cuda()
            # c0 = torch.randn(self.num_layers*self.direction,batch_size,self.rnn_hidden_dim).cuda()
            # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
            # rnn_out,(hn,cn) = self.rnn(cnn_output,(h0,c0))
            rnn_out, _ = self.rnn(cnn_output)
            # rnn_out,(hn,cn) = self.rnn(cnn_output.double(),(h0.double(),c0.double()))
        else:
            # h0 = torch.randn(self.num_layers*self.direction,batch_size,self.rnn_hidden_dim).cuda()
            rnn_out, _ = self.rnn(cnn_output)
            # rnn_out,hn = self.rnn(cnn_output.double(),h0.double())
        # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
        output = self.fc2(nn.functional.relu(rnn_out[:, -1, :]))
        # output = self.softmax(output)
        return output
