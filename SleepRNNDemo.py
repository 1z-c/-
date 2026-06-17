import torch
import torch.nn as nn
import numpy as np
import os
import signal
import sys

# ================================================================
# 教学配置：通过修改这个变量，自由切换不同的循环神经网络进行对比
# 可选值: 'RNN' (标准循环神经网络), 'GRU' (门控循环单元), 'LSTM' (长短期记忆网络)
# ================================================================
MODEL_TYPE = 'LSTM'  # 可改为 'RNN' 或 'GRU'

# 动态生成文件名，防止不同模型的权重互相覆盖
CHECKPOINT_PATH = f'sleep_{MODEL_TYPE.lower()}_checkpoint.pth'
FINAL_WEIGHTS_PATH = f'sleep_{MODEL_TYPE.lower()}_weights.pth'

# 训练配置
TOTAL_EPOCHS = 200  # 问题5/6: 训练200个Epoch对比RNN和LSTM
HIDDEN_SIZE = 32  # 问题8: 可改为4或128对比效果
LEARNING_RATE = 0.01


# ========== 1. 信号捕获与安全性配置（针对共享显卡） ==========
def receive_signal(signum, frame):
    print(f"\n[警告] 收到资源回收信号 (Signal: {signum})! 正在紧急保存当前进度...")
    global model, optimizer, epoch, loss
    save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)
    print(f"[退出] {MODEL_TYPE} 模型的进度已安全保存，程序优雅退出。")
    sys.exit(0)


signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)


def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"> 检查点已保存至: {path}")


# ========== 2. 核心教学知识点：多网络统一实现 ==========
class SleepRNNDemo(nn.Module):
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SleepRNNDemo, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 问题1: 三种RNN变体的API实例化
        if self.cell_type == 'RNN':
            # (1) 标准RNN - 存在梯度消失问题
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'GRU':
            # (2) GRU - 参数更少，计算更快
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'LSTM':
            # (3) LSTM - 长期记忆能力最强
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("未知的网络类型！请选择'RNN','GRU'或'LSTM'")

        self.fc = nn.Linear(hidden_size, output_size)

    # 问题2: 修正后的前向传播（正确处理LSTM的双状态）
    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        # 初始化隐藏状态 h0（所有模型都有）
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if self.cell_type == 'LSTM':
            # LSTM 需要额外初始化细胞状态 c0
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            # RNN 和 GRU 只需要 h0
            out, hn = self.rnn_core(x, h0)

        return self.fc(out)


# ========== 3. 模拟睡眠波形数据生成 ==========
def generate_mock_data(device):
    # 修改：增加数据长度和复杂度，更好地展示梯度消失问题
    t = np.linspace(0, 50, 500)
    # 模拟更复杂的复合睡眠脑电波（包含基础慢波、快波和噪声）
    data = np.sin(t) + 0.5 * np.cos(t * 2.5) + 0.3 * np.sin(t * 5) + np.random.normal(0, 0.05, t.shape)
    x = torch.tensor(data[:-1], dtype=torch.float32).view(1, -1, 1).to(device)
    y = torch.tensor(data[1:], dtype=torch.float32).view(1, -1, 1).to(device)
    return x, y


# ========== 4. 主训练逻辑 ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========= 教学实验: 当前正在训练 【{MODEL_TYPE}】 模型 ==========")
    print(f"运行设备: {device}")
    print(f"隐藏层大小: {HIDDEN_SIZE}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 实例化指定的模型
    model = SleepRNNDemo(cell_type=MODEL_TYPE, hidden_size=HIDDEN_SIZE).to(device)

    # 问题7: 打印参数量（用于对比RNN/GRU/LSTM的参数规模）
    total_params = sum(p.numel() for p in model.parameters())
    print(f"【问题7】总参数量: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_epoch = 0
    loss = torch.tensor(0.0)

    # 断点续训自动绑定当前模型类型
    if os.path.exists(CHECKPOINT_PATH):
        print(f"发现【{MODEL_TYPE}】的历史训练记录，正在恢复...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"成功恢复！从第 {start_epoch} 个 Epoch 继续。")

    x, y = generate_mock_data(device)

    try:
        for epoch in range(start_epoch, TOTAL_EPOCHS):
            model.train()
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'[{MODEL_TYPE}] Epoch [{epoch + 1}/{TOTAL_EPOCHS}], Loss: {loss.item():.6f}')
                save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n[成功] {MODEL_TYPE} 模型训练完成！")

        # 导出特定模型的轻量化参数
        torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
        print(f"部署权重已保存至: {FINAL_WEIGHTS_PATH}, 请下载至本地。")

        # 训练完成后删除检查点文件
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    except Exception as e:
        print(f"训练发生意外: {e}")
        save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)