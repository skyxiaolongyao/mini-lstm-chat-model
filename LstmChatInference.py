import pandas as pd
import torch

from ChatDataset import build_vocab, ChatDataset
from LstmChatModel import ChatModel
from TrainDataFrame import get_train_chat_data

if __name__ == '__main__':
    # 获取训练数据
    train_df_list = get_train_chat_data()

    # 定义一些超参数
    # 词汇表大小
    vocab_size = 10000
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
    # 从文件中加载模型
    model.load_state_dict(torch.load('MiniLstmChatModel.pth', weights_only = True))
    # 设置模型为评估模式 (对于推理很重要)
    model.eval()
    # 构建词汇表
    vocab = build_vocab(train_df_list, vocab_size)
    # 从训练数据里拿一个或多个对话记录
    test_data = {
        'input_text': ['你是一个怎样的语言模型？'],
        'target_text': ['聊天模型，训练过的对话回复准确率能达到99%，暂时不支持其它大模型的训练数据和其它格式的数据。']
    }
    test_df_list = pd.DataFrame(test_data)
    test_dataset = ChatDataset(test_df_list)
    with torch.no_grad():
        for i in range(len(test_dataset)):
            input_text, target_text = test_dataset[i]
            input_ids = [vocab.stoi[word] for word in input_text]
            output = model(torch.tensor(input_ids))
            predicted_ids = torch.argmax(output, dim=1).tolist()
            try:
                predicted_text = ' '.join([vocab.itos[id] for id in predicted_ids if 0 <= id < len(vocab.itos)])
            except IndexError as e:
                print(f"错误: {e}")
                print("请检查 predicted_ids 列表和 vocab.itos 列表，是否存在超出范围的索引。")

            print(f'问题: {input_text}')
            print(f'正确回答: {target_text}')
            print(f'预测回答: {predicted_text}')