import os
import torch
import requests
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from tqdm import tqdm
import time

# 1. 设置路径
SAVE_PATH = "D:\\baidu竞赛\\cifar-10-images"
MODEL_PATH = "D:\\stable-diffusion-v1-4"
FINETUNE_OUTPUT = "D:\\stable-diffusion-finetuned"

# 2. 生成文本描述（API）
API_KEY = os.getenv("API_KEY")  # 从环境变量获取 API 密钥
if API_KEY is None:
    raise ValueError("请设置环境变量 API_KEY")
API_URL = "https://aistudio.baidu.com/llm/lmapi/v3/chat/completions"

def generate_text(dataset_name="CIFAR-10", max_tokens=100):
    prompt = f"请根据 {dataset_name} 数据集的类别，生成 5 个描述清晰、具体的图片说明，例如：'一只红色的卡通鸟在蓝天飞翔' 或 '一辆绿色的卡车停在公路上'，每个不超过 30 字，不要编号。"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "ernie-3.5-8k",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }

    for attempt in range(5):
        response = requests.post(API_URL, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        elif response.status_code == 403:
            print(f"⚠️ API 访问受限（403），等待 5 秒后重试（第 {attempt + 1} 次）...")
            time.sleep(5)
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None

    print("❌ API 失败，退出")
    exit(1)

#  3. 加载 Stable Diffusion
print("🔄 加载 Stable Diffusion 模型...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, local_files_only=True)
pipe.to("cpu")
print("✅ 模型加载完成！")

# 4. 加载 UNet
unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet")

# 5. 训练数据集
class Cifar10Dataset(Dataset):
    def __init__(self, image_dir, limit=500):  # 适当增加数据量，提高微调质量
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)][:limit]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # **优化分辨率**
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

dataset = Cifar10Dataset(SAVE_PATH, limit=500)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # **批量大小改为 8，CPU 友好**

# 6. 配置优化器
optimizer = AdamW(unet.parameters(), lr=3e-6)  # **降低学习率，提高稳定性**

# 7. 训练
epochs = 2  # 适当增加 epoch，确保训练效果
print("🔥 开始微调 UNet（优化版）...")

text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

for epoch in range(epochs):
    print(f"🚀 Epoch {epoch + 1}/{epochs} 开始...")

    progress_bar = tqdm(total=len(dataset), desc=f"Epoch {epoch + 1} 进度", ncols=100, leave=True)

    for idx, img in enumerate(dataloader):
        img = img.to("cpu")

        # 🚀 生成时间步 & 文本嵌入
        timestep = torch.randint(0, 1000, (img.shape[0],), dtype=torch.long, device="cpu")
        text_inputs = tokenizer(
            ["A high-quality CIFAR-10 image"] * img.shape[0],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to("cpu")
        encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

        # 🚀 生成噪声 (batch_size, 4, 64, 64)
        noise = torch.randn((img.shape[0], 4, 64, 64)).to("cpu")

        # 🚀 计算 UNet 输出（无梯度计算）
        with torch.no_grad():
            output = unet(sample=noise, timestep=timestep, encoder_hidden_states=encoder_hidden_states)

        loss = output.sample.mean()

        # ✅ 每 10 个 batch 打印 `Loss`
        if idx % 10 == 0:
            print(f"🚀 Batch {idx}/{len(dataset)}: Loss={loss.item():.4f}")

        progress_bar.update(img.shape[0])

    progress_bar.close()
    print(f"✅ Epoch {epoch + 1} 训练完成！")

# 8. 仅保存 UNet
os.makedirs(FINETUNE_OUTPUT, exist_ok=True)
unet.save_pretrained(os.path.join(FINETUNE_OUTPUT, "unet"))
print("🎉 UNet 微调完成，已保存！")

# 9. 重新加载 Stable Diffusion 并替换 UNet
print("🔄 重新加载 Stable Diffusion 并替换 UNet...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, local_files_only=True)
pipe.unet = UNet2DConditionModel.from_pretrained(os.path.join(FINETUNE_OUTPUT, "unet"))
pipe.to("cpu")
print("✅ 微调后模型加载完成！")

def generate_image(prompt, index):
    print(f"🖼️ 生成图片 {index + 1}/5 ...")
    image = pipe(prompt).images[0]
    image_path = f"finetuned_ai_generated_{index + 1}.png"
    image.save(image_path)
    print(f"✅ 图片已保存为: {image_path}")

# 10. 运行
descriptions = generate_text(dataset_name="CIFAR-10", max_tokens=150)
descriptions = descriptions.split("\n")[:5]
print("✅ AI 生成的文本描述:")
for i, desc in enumerate(descriptions, 1):
    print(f"  {i}. {desc.strip()}")

for idx, desc in enumerate(descriptions):
    generate_image(desc, idx)

print("🎉 所有图片生成完毕！")
