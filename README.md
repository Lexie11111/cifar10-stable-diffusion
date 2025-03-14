# cifar10-stable-diffusion
基于 Stable Diffusion 的文本驱动图像生成与微调 本项目使用 Stable Diffusion 结合 CIFAR-10 数据集，通过 UNet 微调 生成更符合小尺寸图像风格的 AI 生成图片。核心流程包括 文本生成（ERNIE 3.5 API） → 模型微调 → 文本驱动的图像生成，适用于 AI 生成艺术、数据增强、图像风格迁移 等应用场景。
🖼️ CIFAR-10 Stable Diffusion 生成图像项目
📌 项目简介
本项目结合 Stable Diffusion 和 CIFAR-10 数据集，实现文本驱动的图像生成。
主要流程：
1. 文本描述生成：调用百度 ERNIE 3.5 生成符合 CIFAR-10 数据特点的文本描述。
2. 模型微调：对 Stable Diffusion 进行轻量级 UNet 微调，使其更适应 CIFAR-10 风格的小尺寸图像。
3. 图像生成：使用 微调后的模型，根据 AI 生成的文本创建图像。
   
📂 数据集下载
由于 CIFAR-10 数据集较大，建议用户手动下载：
-官方地址：https://www.cs.toronto.edu/~kriz/cifar.html
-下载方式：
bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
-使用方法：
--下载后解压，并将 cifar-10-images/ 放入 项目根目录。

🛠️ 预训练模型下载
由于 Stable Diffusion v1.4 预训练模型体积较大，建议用户手动下载：
-下载地址：https://huggingface.co/CompVis/stable-diffusion-v1-4
-下载方式：
bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
-使用方法：
--下载后解压，并将 stable-diffusion-v1-4/ 放入项目根目录。

🔧 运行环境 & 依赖
-环境要求
--平台：AI Studio / 本地 Python 环境
--Python 版本：Python 3.7+
--深度学习框架：
---PaddlePaddle（适用于 AI Studio）
---PyTorch（适用于 本地运行）

-依赖安装
请先安装以下 Python 依赖项：
bash
pip install torch torchvision diffusers transformers pillow requests tqdm

🚀 运行步骤
 运行模型微调并生成图像
bash
python train.py
该命令将执行以下操作：
--对 Stable Diffusion 的 UNet 进行微调，使其更适用于 CIFAR-10 风格的小尺寸图像。
--调用百度 API 生成符合 CIFAR-10 数据特点的文本描述。
--使用微调后的 Stable Diffusion 生成相应的图像 并保存至本地。
✅ 运行完成后，生成的图像将保存在本地目录中。

🎨 生成结果展示
以下是 CIFAR-10 Stable Diffusion 生成的 五张示例图片：

生成文本	                                   生成图片
1. 一只黄色小狗在草地上欢快奔跑	             https://github.com/user-attachments/assets/3ebee383-b91c-4873-b349-4da9c9f08f9a                
2. 一架蓝白相间的飞机翱翔天际                https://github.com/user-attachments/assets/fd903c1f-b9c8-4391-bdf4-f61486631a61	     
3. 一匹黑色骏马在田野上疾驰	                 https://github.com/user-attachments/assets/852363ab-f04b-476d-b830-c6c42e13eb88
4. 一艘蓝色大船航行在平静海面	               https://github.com/user-attachments/assets/2b19e6fa-1a5e-4ef4-9c01-50400111efa8
5. 一台红色汽车穿梭在城市街道                 https://github.com/user-attachments/assets/16481588-d811-4287-8283-064ae1dcded1

📜 代码结构
cifar10-stable-diffusion/
│── .gitignore
│── LICENSE
│── README.md
│── requirements.txt
│── train.py

📢 贡献 & 反馈
欢迎大家 Star ⭐ 和 Fork 本项目！
如有任何问题，可以在 Issues 中提交反馈。

📌 作者：@Lexie11111

🔗 相关资源
CIFAR-10 数据集：https://www.cs.toronto.edu/~kriz/cifar.html
Stable Diffusion v1.4：https://huggingface.co/CompVis/stable-diffusion-v1-4
📢 喜欢这个项目？欢迎 ⭐Star & Fork！ 🎉
