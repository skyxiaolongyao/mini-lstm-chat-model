# mini-lstm-chat-model
mini-lstm-chat-model 是一个简单易用的开源项目，专为想要从零开始构建小显存高精度对话机器人的开发者设计。采用轻量级的 LSTM 模型，并提供清晰的代码结构和说明文档，即使是初学者也能轻松上手。训练过的对话回复准确率能达到99%，暂时不支持其它大模型的训练数据和其它格式的数据。

# 为什么要从零开始构建
初学者最先接触的通常都是知名公司（如谷歌、微软、Meta 等）发布的，或者基于这些公司模型构建的，预训练模型。不管是微软的 phi-3 系列，还是 google 的 gemma2，又或是 meta 的 llama3，即使是它们的较小版本，也通常需要至少4GB甚至更多的显存才能加载。 这主要是因为这些模型的参数量巨大，需要大量的内存来存储模型权重和激活值。 而且，正如你指出的，在许多实际应用场景中，模型中大部分参数对于特定任务来说可能并不重要，或者冗余度很高。
这时候有的人就会思考，那些模型是怎样调出来的，于是就会去查找，虽然一些模型开源了源代码，但解读这些代码仍然非常困难。
minimind 虽然说2gb 显存加上三小时左右的训练就能出一个模型，但是这种低资源的训练必然会影响模型的准确率。
结合以上的种种问题，mini-lstm-chat-model 的优势就体现出来了，不需要太长的时间，也不需要太多的显存，准确率也很高。
mini-lstm-chat-model 的劣势就是无法做到智能，也可以说它一根筋，但它的优势可以让我们体验从零开始构建小显存高精度对话模型的乐趣。

# 以 Anaconda 环境为例
使用 Anaconda 搭建 Python 开发环境是一种便捷且高效的方法，尤其适合数据科学、机器学习等领域。Anaconda 是一个开源的 Python 发行版，它预装了许多常用的科学计算库和工具，例如 NumPy、Pandas、Scikit-learn、TensorFlow 等，免去了手动安装和配置的繁琐步骤，避免了版本冲突等问题。

## 创建环境和安装所需库
创建环境：  
conda create -n lstm-chat python=3.9  
conda activate lstm-chat  

安装所需库：  
conda install pytorch torchvision torchaudio cpuonly -c pytorch（本例子 cpu 版）  
conda install torchtext=0.6.0  
conda install pandas=2.2  
注意：  
cund 11.8：  
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch

# 开发工具
PyCharm，看个人喜好，用自己习惯的开发工具就行

# 文件说明
启动文件  
1、train.py（训练 lstm 聊天模型）  
2、LstmChatInference.py（lstm 聊天模型推理）  

文件详情  
train.py（训练 lstm 聊天模型）  
描述：  
使用优化的数据加载和配置方法训练了一个 LSTM 聊天机器人。模型状态，包括权重、优化器状态和超参数，使用 .pth（MiniLstmChatModel.pth） 格式高效保存和加载，方便模型的管理。

LstmChatInference.py（lstm 聊天模型推理）   
描述：  
基于 LSTM 的预训练对话模型（MiniLstmChatModel.pth）被从文件中加载，并利用其文本编码和解码完成单轮人机对话交互，实现了流畅自然的文本生成。

LstmChatModel.py（定义模型类）  
描述：  
采用模块化设计，分别实现LSTM聊天机器人模型的核心组件、训练流程和评估指标，方便用户根据需求进行定制和扩展，提升了代码的可维护性和可重用性。

ChatDataset.py（定义模型类）  
描述：  
提供了经过优化的预处理资源，包括高效的数据集管理类和针对聊天数据的定制化停用词表，为提高自然语言处理任务效率提供了保障。

TrainDataFrame.py（自定义对话数据）  
描述：  
高质量的自定义对话训练数据，以DataFrame格式存储，为模型训练提供了可靠且特定领域的训练样本，有效提升了模型的性能和应用效果。

MiniLstmChatModel.pth（存储模型参数文件）  
描述：  
对话模型训练结束后，模型参数（包括权重、偏置以及优化器状态等）被序列化为.pth 文件，该文件包含了模型的完整状态信息，以便后续进行模型加载、推理和微调。

# 交流与赞助
本项目旨在为社区提供有价值的资源。如果您从中获益，欢迎在代码仓库给予 Star，以促进项目的可见性和影响力。 您的赞助将直接支持项目维护和进一步的开发。

# 贡献代码
欢迎各路英雄豪杰 PR 代码 请提交到 dev 开发分支 统一测试发版

# gitee 地址
https://gitee.com/williammy/mini-lstm-chat-model.git

# License
[MIT License](LICENSE)