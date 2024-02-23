# 导入所需的库
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn.functional as F

# 设定使用的设备（GPU或CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 载入预训练的ResNet18模型
model = models.resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(device)

# 读取imagenet类别索引文件
def load_idx_to_labels(csv_path):
    df = pd.read_csv(csv_path)
    idx_to_labels = {row['ID']: [row['wordnet'], row['Chinese']] for idx, row in df.iterrows()}
    return idx_to_labels

# 图像预处理步骤
test_transform = transforms.Compose([
    transforms.Resize(256),  # 缩放图像使其最短边为256像素
    transforms.CenterCrop(224),  # 从图像中心裁剪224x224的区域
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
])

# 图像路径
img_path = 'test_img/computer.jpg'  # 测试图像的路径

# 加载并预处理图像
img_pil = Image.open(img_path)
input_img = test_transform(img_pil).unsqueeze(0).to(device)

# 执行模型预测
pred_logits = model(input_img)
pred_softmax = F.softmax(pred_logits, dim=1)
top_n = torch.topk(pred_softmax, 5)  # 获取置信度最高的5个预测结果
pred_ids = top_n[1].cpu().detach().numpy().squeeze()
confs = top_n[0].cpu().detach().numpy().squeeze()

# 载入类别标签
idx_to_labels = load_idx_to_labels('imagenet_class_index.csv')

# 在图像上标注预测结果
font = ImageFont.truetype('SimHei.ttf', 20)  # 指定中文字体和字号
draw = ImageDraw.Draw(img_pil)
for i in range(len(pred_ids)):
    class_name = idx_to_labels[pred_ids[i]][1]  # 获取类别名称
    confidence = confs[i] * 100  # 转换置信度为百分比
    text = '{:<15} {:>.2f}%'.format(class_name, confidence)  # 格式化文本
    draw.text((10, 10 + 30 * i), text, font=font, fill=(255, 0, 0))  # 在图像上绘制文本

# 显示图像
img_pil.show()
img_pil.save('output/img_pred.jpg')