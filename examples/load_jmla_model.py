import torch
from pathlib import Path

model_path = Path("/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_4-step_8639-allstep_60000.pth")

# 加载模型
checkpoint = torch.load(model_path, map_location='cpu')

# 查看模型信息
print("模型键值:")
for key in checkpoint.keys():
    print(f"  {key}")

# 如果有模型状态
if 'model_state_dict' in checkpoint:
    print(f"\n模型参数数量: {len(checkpoint['model_state_dict'])}")
elif 'state_dict' in checkpoint:
    print(f"\n模型参数数量: {len(checkpoint['state_dict'])}")
else:
    print(f"\n顶层键数量: {len(checkpoint)}")
