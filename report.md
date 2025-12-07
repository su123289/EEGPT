三、论文总结：

1.EEGPT用于EEG信号通用可靠表示的预训练变换器这篇论文遇到了以下问题：

a.信号本身特性限制：EEG 信号信噪比（SNR）极低、个体间变异性大、通道配置不匹配，导致难以提取鲁棒且通用的特征表征；

b.现有模型适配性差：传统掩码自编码器（常用于 NLP/CV）难以从低 SNR 的 EEG 信号中学习高质量抽象特征，易受噪声干扰；

c.设备兼容性问题：不同 EEG 采集设备的采样率、电极位置不一致，导致模型难以跨设备复用，鲁棒性和扩展性不足；

d.计算与适配效率低：传统模型将空间和时间信息耦合处理，计算复杂度高，且对 BCI 多样化应用场景的灵活性适配不足。

2.解决问题的核心创新点，论文提出 EEGPT（1000 万 + 参数预训练 Transformer 模型），核心创新点围绕 “通用、鲁棒、高效” 三大目标设计，共 4 个关键创新：

创新点a.双自监督学习方法（核心突破）—— 解决低 SNR 导致的特征质量差问题。设计 “时空表示对齐 + 基于掩码的重建” 双分支自监督预训练任务，而非依赖单一掩码重建。时空表示对齐：构建基于 “高 SNR + 丰富语义的 EEG 表示” 的自监督任务，而非直接使用原始低 SNR 信号；引入编码器（ENC）、预测器（PRED）和动量编码器（MENC） ：编码器提取掩码部分的特征，预测器结合旋转位置编码（引入相对时序信息）预测完整特征，动量编码器实时更新并输出全局特征，通过均方误差（MSE）对齐预测特征与动量编码器特征，强制模型学习全局一致、鲁棒的时空关联。基于掩码的重建：对 EEG 信号进行 50% 时间掩码 + 80% 通道掩码，重建器（REC）利用编码器的掩码特征、预测器的完整特征及位置信息，通过 “跳跃连接” 保留原始特征结构，重建未掩码部分的信号；损失函数为两分支损失之和，同时优化特征质量和信号重建精度。

创新点b.分层结构设计 —— 解决时空信息处理效率低、灵活性不足的问题。首先通过局部时空嵌入将 EEG 信号拆分为 “通道 - 时间” 补丁（如 58 通道 ×16 个 250ms 时间窗补丁）；编码器专注提取每个时间片段内的空间特征（通道间关联），预测器和动量编码器专注捕捉时间特征（时序关联）；summary token（类似 [CLS] token）汇总同一时间补丁的全局信息，实现时空信息的高效解耦与融合。

创新点c.局部时空嵌入方法 —— 解决通道不匹配、设备兼容性差的问题。设计基于 “通道映射 + 补丁嵌入” 的局部时空嵌入模块：构建Codex book（可学习的通道嵌入字典），包含所有电极通道的嵌入向量，并建立 “通道名称→嵌入向量” 的映射关系；将 EEG 信号在时空维度分割为无重叠补丁，对每个补丁进行线性嵌入，同时融合对应的通道嵌入信息，生成统一维度的 token。

创新点d.混合多任务预训练 + 线性探测策略 —— 解决通用性不足、下游适配成本高的问题。混合多任务数据集预训练：基于 PhysioMI（运动想象 / 执行）、HGD（运动想象）、TSU（SSVEP）、SEED（情绪识别）、M3CV（多范式）等多源数据集预训练，模型参数规模达 1000 万 - 1.01 亿，充分学习跨任务、跨范式的通用特征；下游任务线性探测（Linear-probing）冻结预训练编码器参数，仅训练 “自适应空间滤波器（1×1 卷积）+ 线性分类头”：自适应空间滤波器用于对齐下游数据与预训练模型的通道分布；线性分类头将编码器输出的特征映射为任务标签，避免全微调导致的过拟合和计算消耗。

3.论文流程图及创新点位置
![](https://github.com/su123289/EEGPT/blob/main/images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20251207152426_486_2.png)
![](https://github.com/su123289/EEGPT/blob/main/images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20251207153256_489_2.png)

四、论文公式和程序代码文件名 行数对照表

公式 (1)  传统掩码自编码器损失（输入掩码→编码器→解码器重构）    modeling_pretraining.py    代码行数276-290

公式 (2)  双自监督损失（新增时空表示对齐分支）                 modeling_pretraining.py    代码行数400-500

公式 (3)  编码器输出enc_j（处理掩码 token，整合空间信息）      modeling_pretraining.py    代码行数680-720

公式 (4)  预测器输出pred_j（结合旋转位置嵌入 RoPE）            modeling_pretraining.py    代码行数450-480

公式 (5)  动量编码器输出menc_j（参数更新因子 τ=0.01）          engine_pretraining.py      代码行数150-160

公式 (6)  时空对齐损失L_A（MSE + 层归一化 LN）                 engine_pretraining.py      代码行数85-90

公式 (7)  重构器输出rec_{u,t}（编码器 - 重构器跳跃连接）        modeling_pretraining.py    代码行数360-390

公式 (8)  掩码重构损失L_R（MSE）                              engine_pretraining.py      代码行数90-95

公式 (9)  总预训练损失L = L_A + L_R                           engine_pretraining.py      代码行数95-100

公式 (10) 局部时空嵌入（分块p_{i,j}+ 通道嵌入Embed(p_{i,j})）   modeling_pretraining.py    代码行数315-320；630

公式 (11) 通道嵌入映射ℜ: c_i → ζ_i（灵活适配多数据集通道）      modeling_pretraining.py    代码行数630-635

五、安装说明, 数据集准备，依赖说明，运行配置命令行等。

1.安装说明：

a.环境要求：Python 3.8-3.10（推荐 3.9）；CUDA 11.3+（支持 GPU 加速，单卡 / 多卡均可，论文使用 8 张 NVIDIA 3090）；PyTorch 1.10.0+；PyTorch Lightning 1.6.0+；

b.安装步骤:首先下载代码仓库git clone https://github.com/BINE022/EEGPT.git，cd EEGPT。然后创建虚拟环境conda create -n eegpt python=3.9，conda activate eegpt。最后安装依赖包pip install requirements,有不兼容的库需要先卸载然后找到兼容的库安装。

2.数据集准备：论文里有预处理数据集PhysioMI、HGD、TSU、SEED、M3CV和下游任务数据集BCIC-2A、BCIC-2B、Sleep-EDFx、KaggleERN、PhysioP300、TUAB、TUEV。这里我准备了BCIC-2A数据集，下载地址在https://www.bbci.de/competition/iv/#datasets。

3.核心依赖包功能说明：PyTorch1.10.0+ ——模型构建与张量计算；PyTorch Lightning1.6.0+ ——训练流程管理（分布式训练、日志记录）；MNE1.3.1——脑电图数据读取、滤波、预处理；Braindecode0.7.1——EEG 信号窗口划分、特征提取辅助；Torcheeg1.0.0——通道映射、EEG 专用数据增强；Timm0.6.12——Transformer 模型组件复用（如注意力机制）；Scikit-learn1.2.2——评估指标计算（BAC、F1、Kappa）

4.运行配置命令行：由于预训练成本高，电脑配置等综合原因，仅对下游任务linear_probe_LaBraM_BCIC2A进行复现

a.首先对BCIC2A数据集预处理：原始参数：22 通道、采样率 250Hz、4s / 试次，预处理：0-38Hz 带通滤波（强化运动想象相关脑电成分）；执行 EA 归一化（Euclidean space data alignment），降低被试间差异；通道映射：通过自适应空间滤波器，将 22 通道映射为模型支持的 10-20 系统标准通道。命令:cd downstream/Data_process，python process_function.py。预处理后会生成BCIC_2a_0_38HZ文件夹。

b.预训练模型LaBraM 下载与放置：downstream/Modules/LaBraM/labram-base.pth。进行下游任务linear_probe_LaBraM_BCIC2A，运行命令cd downstream，python linear_probe_LaBraM_BCIC2A.py，受试者一共9人，每人100轮。

六、测试/运行结果截图
![](https://github.com/su123289/EEGPT/blob/main/images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20251206210431_482_2.png)
![](https://github.com/su123289/EEGPT/blob/main/images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20251207175207_491_2.png)
![](https://github.com/su123289/EEGPT/blob/main/images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20251207175236_492_2.png)
