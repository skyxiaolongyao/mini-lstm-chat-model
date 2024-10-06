import torch
from torch import nn

from ChatDataset import build_vocab, ChatDataset
from LstmChatModel import ChatModel, train
from TrainDataFrame import get_train_chat_data


if __name__ == '__main__':
    """
    主函数，用于训练聊天机器人。
    """
    # 获取训练数据
    train_df_list = get_train_chat_data()
    # 训练过的对话才能准确回复，所以测试也用训练数据
    test_df_list = get_train_chat_data()
    if len(train_df_list) >= 1:
        # 定义一些超参数
        # 词汇表大小
        vocab_size = 10000

        # 构建词汇表
        vocab = build_vocab(train_df_list, vocab_size)
        # 创建训练数据集
        train_dataset = ChatDataset(train_df_list)
        # 创建测试数据集
        test_dataset = ChatDataset(test_df_list)

        # 定义模型参数
        # 词向量维度
        embedding_dim = 128
        # 隐藏层维度
        hidden_dim = 256
        # 选择设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建模型
        model = ChatModel(vocab_size, embedding_dim, hidden_dim)
        # 将模型移动到设备上
        model = model.to(device)
        # 创建优化器和损失函数
        # 使用Adam优化器
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        # 使用交叉熵损失函数
        loss_fn = nn.CrossEntropyLoss()
        # 训练轮数
        epochs = 20
        train(model, train_dataset, optimizer, loss_fn, epochs, vocab, test_dataset)