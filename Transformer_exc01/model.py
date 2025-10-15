import torch

from mask import mask_pad, mask_tril
from util import MultiHead, PositionEmbedding, FullyConnectedOutput

# 编码器层
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead()  # 多头注意力计算层
        self.fc = FullyConnectedOutput()  # 全连接输出层

    def forward(self, x, mask):
        # 计算自注意力（Q、K、V矩阵相同，全是x），维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)

        return out

# 编码器
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 完整的编码器包含了3个编码器层
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x

# 解码器层
class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 多头注意力层
        self.mh1 = MultiHead()
        self.mh2 = MultiHead()

        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算y的自注意力,维度不变（用mask_tril_y直线遮盖t时刻及其之后的数据）
        # [b, 50, 32] -> [b, 50, 32]
        y = self.mh1(y, y, y, mask_tril_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.mh2(y, x, x, mask_pad_x)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.fc(y)

        return y

# 解码器
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(y, x, mask_pad_x, mask_tril_y)
        y = self.layer_3(y, x, mask_pad_x, mask_tril_y)
        return y


# 主模型
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = PositionEmbedding()  # 位置编码层
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()  # 编码器
        self.decoder = Decoder()  # 解码器
        self.fc_out = torch.nn.Linear(32,39)  # 全连接输出层

    def forward(self, x, y):
        # [b, 1, 50, 50]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        # 编码,添加位置信息
        # x = [b, 50] -> [b, 50, 32]
        # y = [b, 50] -> [b, 50, 32]
        x, y = self.embed_x(x), self.embed_y(y)

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x, mask_pad_x)

        # 解码层计算
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出
        # [b, 50, 32] -> [b, 50, 39]
        y = self.fc_out(y)

        return y
