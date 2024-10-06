import pandas as pd

# 自定义对话数据
def get_train_chat_data():
    """
    获取训练对话数据。

    Returns:
        pd.DataFrame: 包含问题和答案的 Pandas DataFrame。
    """
    train_chat_data = {
        'input_text': [
            '你可以自定义训练数据吗？',
            '你是一个怎样的语言模型？',
            '能用其它 AI 一样的格式问你问题吗？',
            '你好',
            '今天天气怎么样',
            '你叫什么名字',
            '再见',
            '你多大年纪',
            '你来自哪里',
            '训练数据从哪里来'
        ],
        'target_text': [
            '是的',
            '聊天模型，训练过的对话回复准确率能达到99%，暂时不支持其它大模型的训练数据和其它格式的数据。',
            '不能，我的思维比较单一，只能回答训练过的对话，没有训练过的对话回复会是胡言乱语。',
            '你好呀',
            '今天天气晴朗',
            '我叫 mini-lstm-chat',
            '再见',
            '60',
            '我来自数字世界',
            'DataFrame'
        ]
    }
    train_chat_df = pd.DataFrame(train_chat_data)
    return train_chat_df