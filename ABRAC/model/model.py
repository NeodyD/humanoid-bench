import torch
import torch.nn as nn
class FallPredictionModel(nn.Module):
    def init(self):
        super(FallPredictionModel, self).__init__()
        
        # 视觉特征提取（CNN）
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 输入：3通道图像
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128)  # 假设输入图像64x64，调整为128维
        )
        
        # IMU和关节数据处理（LSTM）
        self.imu_lstm = nn.LSTM(input_size=6, hidden_size=64, batch_first=True)  # IMU: 3加速度+3角速度
        self.joint_lstm = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)  # 假设10个关节
        
        # ZMP处理（前馈网络）
        self.zmp_fc = nn.Sequential(
            nn.Linear(2, 16),  # ZMP: x, y坐标
            nn.ReLU()
        )
        
        # 融合层（这里使用简单拼接）
        self.fusion_fc = nn.Linear(128 + 64 + 32 + 16, 128)  # 240维 -> 128维
        
        # 输出层
        self.fall_judgment = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()  # 跌倒判断：0或1
        )
        self.fall_time = nn.Linear(128, 1)  # 跌倒时间：回归
        self.fall_direction = nn.Sequential(
            nn.Linear(128, 4),  # 假设4个方向：前、后、左、右
            nn.Softmax(dim=1)
        )
    
    def forward(self, visual, imu, joint, zmp):
        # 视觉特征
        visual_out = self.visual_cnn(visual)  # [batch, 128]
        
        # IMU特征
        imu_out, _ = self.imu_lstm(imu)  # [batch, seq_len, 64]
        imu_out = imu_out[:, -1, :]  # 取最后一个时间步 [batch, 64]
        
        # 关节特征
        joint_out, _ = self.joint_lstm(joint)  # [batch, seq_len, 32]
        joint_out = joint_out[:, -1, :]  # 取最后一个时间步 [batch, 32]
        
        # ZMP特征
        zmp_out = self.zmp_fc(zmp)  # [batch, 16]
        
        # 融合层：简单拼接
        fused = torch.cat((visual_out, imu_out, joint_out, zmp_out), dim=1)  # [batch, 240]
        fused = self.fusion_fc(fused)  # [batch, 128]
        
        # 输出
        judgment = self.fall_judgment(fused)  # [batch, 1]
        time = self.fall_time(fused)          # [batch, 1]
        direction = self.fall_direction(fused)  # [batch, 4]
        
        return judgment, time, direction
# 测试模型
model = FallPredictionModel()
visual_input = torch.randn(2, 3, 64, 64)  # batch=2, 3通道, 64x64图像
imu_input = torch.randn(2, 10, 6)         # batch=2, 10时间步, 6维IMU
joint_input = torch.randn(2, 10, 10)      # batch=2, 10时间步, 10维关节
zmp_input = torch.randn(2, 2)             # batch=2, 2维ZMP
judgment, time, direction = model(visual_input, imu_input, joint_input, zmp_input)
print(judgment.shape, time.shape, direction.shape)  # torch.Size([2, 1]) torch.Size([2, 1]) torch.Size([2, 4])