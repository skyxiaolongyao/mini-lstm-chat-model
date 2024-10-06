# 定义数据集类
from collections import Counter

from torchtext.vocab import Vocab

# 数据集类
class ChatDataset:
    """
    用于处理聊天数据的一个数据集类。

    Args:
        df_list (list):  包含聊天数据信息的 DataFrame 列表。

    Attributes:
        data (list): 存储聊天数据信息的 DataFrame 列表。
    """
    def __init__(self, df_list):
        self.data = df_list

    def __len__(self):
        """
        返回数据集的长度。
        Returns:
            int: 数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本。

        Args:
            idx (int): 样本索引。

        Returns:
            tuple: 包含输入文本和目标文本的元组。
        """
        input_text = self.data['input_text']
        target_text = self.data['target_text']
        return input_text, target_text


# 停用词表
def build_vocab(df_list, vocab_size):
    """
    构建词汇表。

    Args:
        df_list (list): 包含聊天数据信息的 DataFrame 列表。
        vocab_size (int): 词汇表的大小。

    Returns:
        Vocab: 构建好的词汇表对象。
    """
    word_frequency = Counter()
    for index, row in df_list.iterrows():
        # 输入文本，可以理解为：问题
        input_text = row['input_text']
        # 目标文本，可以理解为：回答
        target_text = row['target_text']
        # 提取输入和目标文本的单词
        words = [input_text, target_text]
        # 统计词频
        word_frequency.update(words)

    most_common_words = word_frequency.most_common(vocab_size)
    # 创建词汇表对象
    vocab = Vocab(Counter(dict(most_common_words)))
    # 为未知词添加索引
    vocab.stoi['<unk>'] = 1
    return vocab