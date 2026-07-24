import os
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# ==========================================
# 0. 全局超参数设置
# ==========================================
# 相对路径：数据集文件夹放在代码同级目录
# DATA_DIR = os.path.join(os.path.dirname(__file__), "New_Coal_Dataset_AllAug7.2.1")
# Github公开演示使用
DATA_DIR = os.path.join(os.path.dirname(__file__), "demo_synthetic_dataset")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 25
NUM_CLASSES = 4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
CLASS_NAMES = ["0_原生", "1_碎裂", "2_碎粒", "3_糜棱"]


# ==========================================
# 1. 注意力机制模块定义
# ==========================================
class SE_Module(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        self.zpool = ZPool()
        self.conv = BasicConv(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        return self.conv(self.zpool(x))


class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_hw = self.sigmoid(self.hw(x))
        out_hw = x * x_hw

        x_p = x.permute(0, 2, 1, 3).contiguous()
        x_cw = self.sigmoid(self.cw(x_p))
        out_cw = (x_p * x_cw).permute(0, 2, 1, 3).contiguous()

        x_p2 = x.permute(0, 3, 2, 1).contiguous()
        x_hc = self.sigmoid(self.hc(x_p2))
        out_hc = (x_p2 * x_hc).permute(0, 3, 2, 1).contiguous()
        return (out_hw + out_cw + out_hc) / 3


# ==========================================
# 2. ResNet-AHA 网络构建
# ==========================================
class ResNet_AHA(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True, dropout=0.5):
        super(ResNet_AHA, self).__init__()
        if use_pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            print("✅ 已加载 ImageNet 预训练权重，正在迁移至单通道输入...")
        else:
            resnet = models.resnet18(weights=None)
            print("ℹ️ 从头训练，未使用预训练权重。")

        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.se1 = SE_Module(64)

        self.layer2 = resnet.layer2
        self.triplet2 = TripletAttention()

        self.layer3 = resnet.layer3
        self.se3 = SE_Module(256)

        self.layer4 = resnet.layer4
        self.triplet4 = TripletAttention()

        self.avgpool = resnet.avgpool
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)

        x = self.layer2(x)
        x = self.triplet2(x)

        x = self.layer3(x)
        x = self.se3(x)

        x = self.layer4(x)
        x = self.triplet4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ==========================================
# 3. 训练与测试主程序
# ==========================================
if __name__ == '__main__':
    print("📂 正在加载平衡后的地质物理数据集...")

    # 在线数据增强：仅做泛化正则化，防止过拟合
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 直接使用 ImageFolder 读取物理文件夹
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(
        f"📊 数据集分布确认 -> 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")
    # ==========================================
    # 新增：物理数据集统计与显示
    # ==========================================
    print("\n🔍 正在扫描物理文件夹进行最终数据统计...")

    def count_files(directory):
        # 扫描子文件夹中的图片数量
        stats = {}
        for root, dirs, files in os.walk(directory):
            if root == directory:
                continue
            class_name = os.path.basename(root)
            # 只数 .png 文件
            count = len([f for f in files if f.endswith('.png')])
            stats[class_name] = count
        return stats

    train_stats = count_files(os.path.join(DATA_DIR, 'train'))
    val_stats = count_files(os.path.join(DATA_DIR, 'val'))
    test_stats = count_files(os.path.join(DATA_DIR, 'test'))

    print("\n" + "=" * 80)
    print(f" {'煤体结构类别':<15} | {'训练集(已平衡)':<12} | {'验证集':<10} | {'测试集(纯净)':<10} | {'合计':<8}")
    print("-" * 80)

    grand_total = 0
    for name in CLASS_NAMES:
        t_c = train_stats.get(name, 0)
        v_c = val_stats.get(name, 0)
        ts_c = test_stats.get(name, 0)
        row_total = t_c + v_c + ts_c
        grand_total += row_total
        # 针对中文字符对齐处理
        print(f" {name.ljust(17 - len(name.encode('gbk')) + len(name))} | {t_c:<14} | {v_c:<10} | {ts_c:<12} | {row_total:<8}")

    print("-" * 80)
    print(f" {'总计'.ljust(17 - len('总计'.encode('gbk')) + len('总计'))} | {len(train_dataset):<14} | {len(val_dataset):<10} | {len(test_dataset):<12} | {grand_total:<8}")
    print("=" * 80 + "\n")
    model = ResNet_AHA(num_classes=NUM_CLASSES, use_pretrained=True, dropout=0.5).to(DEVICE)

    # ★ 核心改动：由于离线数据已达到平衡，此处将交叉熵权重全部重置为 1.0，不加人工干预
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"🚀 开始在平衡数据集上训练 ResNet-AHA (Device: {DEVICE})...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += inputs.size(0)

        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1:03d}/{MAX_EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⚠️ 触发早停机制 (Patience={PATIENCE})，网络已收敛！")
                break

    # ==========================================
    # 4. 测试集闭卷评估
    # ==========================================
    print(f"\n📊 正在纯净的测试集上进行闭卷评估 (最优权重来自第 {best_epoch} 轮)...")
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 85)
    print(f"        模型名称         |   准确率(Acc)/%   |    精确率(P)    | 召回率(Recall)  |    F1 分数    ")
    print("-" * 85)
    print(
        f" ResNet-AHA (Ours)       |     {acc * 100:5.2f}    % |    {precision:.4f}     |     {recall:.4f}      |    {f1:.4f}    ")
    print("=" * 85)

    print("\n📋 类别详细报告 (关注强构造煤的召回率):")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    print("\n🧮 混淆矩阵 (Confusion Matrix):")
    header = "真实 \\ 预测 | " + " | ".join([f"{name:>8}" for name in CLASS_NAMES])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        name_str = CLASS_NAMES[i]
        padded_name = name_str.rjust(14 - len(name_str.encode('gbk')) + len(name_str))
        row_data = " | ".join([f"{val:>8}" for val in row])
        print(f"{padded_name} | {row_data}")
    print("-" * len(header) + "\n")

    # ---------- 保存模型权重 ----------
    save_path = os.path.join(MODEL_SAVE_DIR, "ResNet_AHA_balanced_best.pth")
    torch.save(best_model_wts, save_path)
    print(f"✅ 最优模型权重已保存至: {save_path}")

    # ---------- 绘制并保存训练曲线 ----------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.axvline(best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves (Balanced Dataset)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.axvline(best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves (Balanced Dataset)')
    plt.grid(True)

    plot_path = os.path.join(MODEL_SAVE_DIR, "balanced_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 训练曲线已保存至: {plot_path}")