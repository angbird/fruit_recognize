import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


model = torch.load('weights/fruit30_pytorch.pth')   # 导入模型
model = model.to('cpu')                             # 将模型数据类型改为CPU，避免输入类型和模型参数类型不一致
model.eval()
scripted_module = torch.jit.script(model)
optimized_scripted_module = optimize_for_mobile(scripted_module)  # 针对移动端进行优化
optimized_scripted_module._save_for_lite_interpreter("weights/fruit30_pytorch_android_new.ptl")

