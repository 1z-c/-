import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import time
import random
import websockets

# =======================
# 工业现场配置
# =======================
MODEL_TYPE = 'LSTM'  # 学生可在此修改为 'RNN' 或 'GRU' 以加载不同模型
WEIGHTS_PATH = f'sleep_{MODEL_TYPE.lower()}_weights.pth'
HOST = "127.0.0.1"
PORT = 8765
HIDDEN_SIZE = 32  # 问题9: 可改为4或128对比效果
SAMPLING_INTERVAL = 0.03  # 采样间隔30ms

# 问题4: 控制是否每次重置隐状态（模拟"失忆"效果）
RESET_HIDDEN_EACH_STEP = False  # 改为True可观察"失忆"现象


# ========== 1. 保持与云端一致的模型结构 ==========
class SleepRNNDemo(nn.Module):
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SleepRNNDemo, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        device = x.device
        batch_size = x.size(0)

        if hidden is None:
            # 初始化隐藏状态
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            if self.cell_type == 'LSTM':
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
                hidden = (h0, c0)
            else:
                hidden = h0

        if self.cell_type == 'LSTM':
            out, hidden = self.rnn_core(x, hidden)
        else:
            out, hidden = self.rnn_core(x, hidden)

        return self.fc(out), hidden


# ========== 2. WebSocket遥测数据流 ==========
async def stream_data(websocket, path=None):
    print(f"\n[终端接入]客户端已连接。开始下发{MODEL_TYPE}遥测数据...")
    print(f"[配置] 重置隐状态: {'是' if RESET_HIDDEN_EACH_STEP else '否'}")

    # 初始化模型
    model = SleepRNNDemo(cell_type=MODEL_TYPE, hidden_size=HIDDEN_SIZE)

    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"[系统]成功加载本地权重文件: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"[警告]未找到{WEIGHTS_PATH}，将使用未经训练的初始权重进行仿真演示。")

    # 生成仿真数据流
    t = np.linspace(0, 100, 3000)
    test_signal = np.sin(t) + 0.5 * np.cos(t * 2.5) + 0.3 * np.sin(t * 5) + np.random.normal(0, 0.05, t.shape)

    # 初始化隐状态（在整个推理过程中保持）
    hidden = None

    try:
        with torch.no_grad():
            for i in range(len(test_signal) - 1):
                start_time = time.time()

                # 问题4: 强行重置隐状态（模拟"失忆"）
                if RESET_HIDDEN_EACH_STEP:
                    # 重置隐状态，丢失所有历史记忆
                    hidden = None

                # 推理 - 输入形状: [batch=1, seq_len=1, input_size=1]
                input_point = torch.tensor([[[test_signal[i]]]], dtype=torch.float32)

                # 前向传播，同时获取新的隐状态
                pred_point, hidden = model(input_point, hidden)
                pred_point = pred_point.item()
                actual_point = test_signal[i + 1]

                # 模拟系统处理延迟
                calc_time = (time.time() - start_time) * 1000
                simulated_latency = calc_time + random.uniform(5.0, 15.0)

                # 打包JSON数据帧
                payload = {
                    "timestamp": time.time() * 1000,
                    "model_type": MODEL_TYPE,
                    "ch1_actual": float(actual_point),
                    "ch2_predict": float(pred_point),
                    "error_abs": abs(float(actual_point) - float(pred_point)),
                    "latency_ms": round(simulated_latency, 2)
                }

                await websocket.send(json.dumps(payload))
                await asyncio.sleep(SAMPLING_INTERVAL)

    except websockets.exceptions.ConnectionClosed:
        print("[断开] 客户端连接已断开")
    except Exception as e:
        print(f"[断开] 客户端连接中断或发生异常: {e}")


async def main():
    async with websockets.serve(stream_data, HOST, PORT):
        print("===========")
        print(f"[SYS] 工业级边缘计算遥测终端已启动")
        print(f"[SYS] 当前挂载计算核心: {MODEL_TYPE} 神经网络")
        print(f"[SYS] 隐藏层大小: {HIDDEN_SIZE}")
        print(f"[SYS] 监听端口: ws://{HOST}:{PORT}")
        print("===========")
        print("等待前端监控面板接入...")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())