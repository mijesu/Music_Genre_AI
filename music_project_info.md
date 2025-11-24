# 音乐AI项目信息

## 项目位置
- 项目文件夹：`/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass/`
- 所有项目材料、脚本和文档都存储在此文件夹中

## 目录结构

### 数据集
- 路径：`/media/mijesu_970/SSD_Data/DataSets`
- 结构：每个数据集有两个子文件夹
  - `Data/` - 原始数据文件
  - `Misc/` - 参考资料和信息
- GTZAN数据集：`/media/mijesu_970/SSD_Data/DataSets/GTZAN/`
  - 10个音乐流派：blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
  - 每个流派约100个音频文件
  - 用途：基础流派分类
- FMA Medium：已下载
  - 25,000个音频文件，16个流派
  - 大小：~22GB
  - 路径：`/media/mijesu_970/SSD_Data/DataSets/FMA/`
  - 用途：改进的流派分类训练数据
- MagnaTagATune：待添加
  - 25,863个音频片段（每个29秒）
  - 188个标签（流派、乐器、情绪、人声等）
  - 大小：~50GB
  - 用途：多标签分类、音乐标注、详细特征分析
- Million Song Dataset (MSD)：待添加
  - 包含100万首歌曲的音频特征和元数据
  - 大小：~280GB压缩
  - 用途：大规模音乐分析、特征对比、元数据增强

### AI模型
- 目录：`/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/`
- 可用模型：
  - ✓ `epoch_20.pth` (330MB) - 早期检查点
  - ✓ `epoch_4-step_8639-allstep_60000.pth` (1.3GB) - 主模型
- 模型类型：Vision Transformer (ViT) 用于音频特征提取
- 作者：MMSelfSup

### 其他可用的预训练音频模型

如需更多预训练模型选择，可考虑：

1. **OpenJMLA** (已下载) ✓
   - 类型：通用音频特征提取器
   - 架构：Vision Transformer (ViT)
   - 参数：86M
   - 用途：音频表示学习

2. **Musicnn**
   - 类型：音乐标注模型
   - 用途：音乐特征提取和标签预测
   - 来源：Music Technology Group

3. **VGGish**
   - 类型：Google音频模型
   - 架构：基于VGG的CNN
   - 用途：通用音频嵌入

4. **CLMR**
   - 类型：对比学习音乐表示
   - 用途：自监督音乐特征学习
   - 特点：无需标签训练

5. **Jukebox**
   - 类型：OpenAI音乐生成模型
   - 用途：音乐生成和理解
   - 特点：大规模Transformer模型

### 待分类音乐
- 路径：`/media/mijesu_970/SSD_Data/Music_TBC`
- 用途：需要进行流派分类的音乐文件

## 已创建的脚本

1. `pytorch_example.py` - PyTorch基础示例
2. `music_genre_classifier.py` - 音乐流派分类脚本
3. `load_jmla_model.py` - JMLA模型加载脚本

## 推荐的Python库

### 音频处理
- librosa - 特征提取、频谱图、节奏、音高
- torchaudio - PyTorch音频加载和转换
- pydub - 音频文件处理和转换

### 音乐信息检索
- essentia - 综合音频分析（节奏、调性、情绪）
- madmom - 节拍跟踪、和弦识别

### 深度学习
- torch - PyTorch神经网络
- nnAudio - GPU音频处理

### 可视化
- matplotlib - 绘制频谱图和波形

## 安装命令

```bash
pip install torch==2.8.0 torchaudio librosa matplotlib
```

## 已安装的库

- ✓ torch 2.8.0
- ✓ torchaudio 2.8.0
- ✓ librosa 0.11.0
- ✓ matplotlib 3.5.1
- ✓ numpy 1.26.4
- ✓ torchvision 0.23.0

注意：essentia需要系统级依赖(fftw3f)，暂时跳过。可以使用librosa进行音频分析。

## 项目方法

使用OpenJMLA预训练模型作为特征提取器，结合GTZAN数据集进行音乐流派分类：
1. 特征提取：使用OpenJMLA提取音频特征
2. 迁移学习：在OpenJMLA基础上添加分类层
3. 训练：使用GTZAN的10个流派数据进行训练

## 下一步

1. 运行 `load_jmla_model.py` 查看模型结构
2. 使用JMLA模型对Music_TBC中的音乐进行分类
3. 根据需要训练或微调模型
