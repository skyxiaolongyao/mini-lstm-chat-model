import torch
from torch import nn
from torchtext.vocab import Vocab


# 定义模型类
class ChatModel(nn.Module):
    """
    聊天模型类，用于构建一个简单的聊天模型。

    该模型使用 LSTM 作为编码器，并使用线性层作为解码器。

    Args:
        vocab_size (int): 词汇表大小。
        embedding_dim (int): 词向量维度。
        hidden_dim (int): LSTM 隐藏层维度。
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        初始化聊天模型。

        Args:
            vocab_size (int): 词汇表大小。
            embedding_dim (int): 词向量维度。
            hidden_dim (int): LSTM 隐藏层维度。
        """
        super(ChatModel, self).__init__()
        # 记录词汇表大小，将 vocab_size 保存为模型属性
        self.vocab_size = vocab_size
        # 定义词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 定义 LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # 定义线性层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入的词向量。

        Returns:
            torch.Tensor: 输出的词向量。
        """
        # 将输入的词向量转换为词嵌入向量
        embedded = self.embedding(x)
        # 将词嵌入向量输入 LSTM 层
        output, (hidden, cell) = self.lstm(embedded)
        # 将 LSTM 的输出输入线性层
        output = self.fc(output)
        # 返回线性层的输出
        return output


# 训练聊天模型
def train(model, train_dataset, optimizer, loss_fn, epochs, vocab, eval_dataset=None):
    """
    训练模型。

    Args:
        model (nn.Module): 模型。
        train_dataset (ChatDataset): 训练数据集。
        optimizer (torch.optim.Optimizer): 优化器。
        loss_fn (nn.Module): 损失函数。
        epochs (int): 训练轮数。
        vocab (Vocab): 词汇表。
        eval_dataset (ChatDataset, optional): 验证数据集。 Defaults to None.
    """
    if eval_dataset:
        # 初始化验证损失
        best_val_loss = float('inf')
        for epoch in range(epochs):
            for i in range(len(train_dataset)):
                input_text, target_text = train_dataset[i]
                # 文本转向量或转one-hot
                # 将输入文本转换为索引列表
                input_ids = [vocab.stoi[word] for word in input_text]
                # 将目标文本转换为索引列表
                target_ids = [vocab.stoi[word] for word in target_text]
                # 将索引列表转换为张量
                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                # 将索引列表转换为张量
                target_tensor = torch.tensor(target_ids, dtype=torch.long)
                # 清空梯度
                optimizer.zero_grad()
                # 前向传播
                output = model(input_tensor)
                # 计算损失
                loss = loss_fn(output, target_tensor)
                # 反向传播
                loss.backward()
                # 更新权重
                optimizer.step()
                print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}")

            # 评估模型
            val_loss = evaluate(model, eval_dataset, loss_fn, vocab)
            print(f"评估准确率: {val_loss}")
            if val_loss < best_val_loss:
                # 更新最佳验证损失
                best_val_loss = val_loss
                # 保存模型
                torch.save(model.state_dict(), 'MiniLstmChatModel.pth')
                # 打印验证损失
                print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            else:
                # 打印验证损失
                print(f"无法生成模型，Epoch {epoch}, Validation Loss: {val_loss:.4f}")
    else:
        for epoch in range(epochs):
            for i in range(len(train_dataset)):
                input_text, target_text = train_dataset[i]
                # 将输入文本转换为索引列表
                input_ids = [vocab.stoi[word] for word in input_text]
                # 将目标文本转换为索引列表
                target_ids = [vocab.stoi[word] for word in target_text]
                # 将索引列表转换为张量
                input_tensor = torch.tensor(input_ids, dtype=torch.long)
                # 将索引列表转换为张量
                target_tensor = torch.tensor(target_ids, dtype=torch.long)
                # 清空梯度
                optimizer.zero_grad()
                # 前向传播
                output = model(input_tensor)
                # 计算损失
                loss = loss_fn(output, target_tensor)
                # 反向传播
                loss.backward()
                # 更新权重
                optimizer.step()


# 评估模型
def evaluate(model, test_dataset, loss_fn, vocab):
    """
    评估模型在测试集上的性能。

    Args:
        model (nn.Module): 模型。
        test_dataset (ChatDataset): 测试数据集。
        loss_fn (nn.Module): 损失函数。
        vocab (Vocab): 词汇表。

    Returns:
        float: 平均损失。
    """
    # 将模型设置为评估模式
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        # 初始化总损失
        total_loss = 0

        # 遍历测试数据集
        for i in range(len(test_dataset)):
            # 获取输入和目标文本
            input_text, target_text = test_dataset[i]
            # 将输入文本转换为索引列表
            input_ids = [vocab.stoi[word] for word in input_text]
            # 将目标文本转换为索引列表
            target_ids = [vocab.stoi[word] for word in target_text]
            # 前向传播
            output = model(torch.tensor(input_ids))
            # 计算损失
            loss = loss_fn(output, torch.tensor(target_ids))
            # 将损失添加到总损失中
            total_loss += loss.item()

        # 计算平均损失
        # 一般来说，平均损失值在 0.1 到 0.5 之间是比较好的。
        # 0.1 以下: 表示模型的性能非常好，预测结果非常准确。
        # 0.1 到 0.5: 表示模型的性能一般，预测结果还算准确。
        # 0.5 以上: 表示模型的性能较差，预测结果不准确。
        avg_loss = total_loss / len(test_dataset)
        print(f'平均损失: {avg_loss}')
        return avg_loss