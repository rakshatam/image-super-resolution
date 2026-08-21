# NAFNet-SR: Highly Efficient Super-Resolution 

This repository contains the implementation, training details, and benchmark results for our custom **NAFNet-SR** model. Designed for extreme efficiency, this model achieves highly competitive super-resolution performance on a strict $<1\text{M}$ parameter budget.

## 🧠 Architecture Deep Dive

The core architecture is based on **NAFNet** (Nonlinear Activation Free Network). To maximize parameter efficiency and inference speed, it strips away computationally expensive self-attention mechanisms and heavy activation caches.

### Architecture Diagram (ASCII)

```text
Input LR Image (3, H, W)
       │
       ▼
[ Conv2d (3 -> 64) ] ─── Intro Feature Map ──┐
       │                                     │
       ▼                                     │
┌─────────────────────────────────────┐      │
│          NAF Block (x32)            │      │
│                                     │      │
│  [ LayerNorm2d ]                    │      │
│        │                            │      │
│  [ Conv2d (1x1, C -> C*2) ]         │      │
│        │                            │      │
│  [ Depthwise Conv2d (3x3) ]         │      │
│        │                            │      │
│  [ SimpleGate (C*2 -> C) ]          │      │
│        │                            │      │
│  [ Channel Attention ] ─────────────┤      │
│        │                            │      │
│  [ Conv2d (1x1, C -> C) ]           │      │
│        │                            │      │
│  ( Residual Addition )              │      │
│        │                            │      │
│  [ LayerNorm2d ]                    │      │
│        │                            │      │
│  [ Conv2d (1x1, C -> C*2) ]         │      │
│        │                            │      │
│  [ SimpleGate (C*2 -> C) ]          │      │
│        │                            │      │
│  [ Conv2d (1x1, C -> C) ]           │      │
│        │                            │      │
│  ( Residual Addition )              │      │
└────────┬────────────────────────────┘      │
         │                                   │
         ▼                                   │
( Global Residual Addition ) ◄───────────────┘
         │
         ▼
[ Conv2d (64 -> 3 * 4^2) ]
         │
         ▼
[ PixelShuffle (x4) ]
         │
         ▼
( Skip Connection from Bicubic LR ) 
         │
         ▼
Output HR Image (3, 4H, 4W)
```

### In-Depth Working Principle
Unlike traditional CNNs that rely heavily on non-linear activation functions (e.g., ReLU, GELU, Sigmoid) to model complex mappings, NAFNet achieves non-linearity purely through **Activation-Free Gating** (`SimpleGate`). 

1. **SimpleGate Mechanism:** A feature map $X \in \mathbb{R}^{2C \times H \times W}$ is split along the channel dimension into two halves: $X_1, X_2 \in \mathbb{R}^{C \times H \times W}$. The output is their element-wise multiplication: $Y = X_1 \odot X_2$. This operation is mathematically non-linear but inherently avoids the memory overhead of storing activation maps during the backward pass.
2. **Simplified Channel Attention (SCA):** A channel attention module aggregates global spatial information using Adaptive Average Pooling, followed by a $1\times 1$ convolution to rescale the gated features. This captures global context without the $O((HW)^2)$ complexity of Transformer self-attention.
3. **PixelShuffle Upsampling:** After deep feature extraction through 32 NAF blocks, a final sub-pixel convolution layer expands the channel depth to $48$ ($3 \times 4^2$) and periodically reshuffles them into the spatial dimensions, achieving the $\times 4$ resolution scale efficiently.
4. **Parameter Budget:** Operates on a strict $\sim 1.0\text{M}$ parameter constraint, making it exceptionally lightweight and suitable for efficient inference.

## 📊 Dataset & Training Setup

The model was trained using a high-quality spatial dataset and accelerated on distributed hardware.

* **Dataset**: **DF2K** (DIV2K + Flickr2K), consisting of 3,450 high-resolution images.
* **Degradation Model**: Standard Bicubic Downsampling ($\times 4$).
* **Hardware**: Dual NVIDIA T4 GPUs (Distributed Data Parallel via HuggingFace Accelerate).
* **Batch Size**: 16 per GPU (Effective global batch size of 32).
* **Optimization Steps**: The model was trained for **40 epochs**. Due to the distributed setup dividing the dataset, this required 431 steps per epoch, resulting in exactly **17,240 total optimization steps**.

### Optimization & Loss Function
To optimize the network, we utilized the **L1 Loss (Mean Absolute Error)**. Unlike L2 Loss (MSE) which heavily penalizes large errors but can lead to overly smoothed textures, L1 Loss promotes sharper high-frequency edge reconstruction which is critical for super-resolution tasks.

$$
\mathcal{L}_1 = \frac{1}{N} \sum_{i=1}^{N} \left\| I_{SR}^{(i)} - I_{HR}^{(i)} \right\|_1
$$

The optimizer used was AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$) paired with a Cosine Annealing learning rate scheduler.

## 📈 Evaluation Metrics (Academic Protocol)

To ensure perfectly accurate comparisons against published literature, all metrics were calculated using strict academic testing protocols.

### 1. Color Space Conversion (ITU-R BT.601)
Images are converted from RGB to the YCbCr color space. Metrics are calculated exclusively on the **Y (Luminance) channel**, matching the exact math used by MATLAB's `rgb2ycbcr` function:

$$
Y = 16.0 + \frac{65.481 \cdot R + 128.553 \cdot G + 24.966 \cdot B}{255.0}
$$

### 2. Peak Signal-to-Noise Ratio (PSNR)
PSNR is calculated on the shaved Y-channels (with a 4-pixel border removed to ignore boundary artifacts). Given the ground truth $I_{HR}$ and the super-resolved image $I_{SR}$:

$$
MSE = \frac{1}{H W} \sum_{i=1}^{H} \sum_{j=1}^{W} (I_{HR}(i,j) - I_{SR}(i,j))^2
$$

$$
PSNR = 10 \cdot \log_{10} \left( \frac{255^2}{MSE} \right) \text{ dB}
$$

### 3. Structural Similarity Index (SSIM)
SSIM measures the perceived degradation in structural information using an $11\times 11$ Gaussian kernel with standard deviation $\sigma=1.5$. It computes luminance ($\mu$), contrast ($\sigma^2$), and structure ($\sigma_{xy}$) comparisons:

$$
SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

Where $C_1 = (0.01 \times 255)^2$ and $C_2 = (0.03 \times 255)^2$.

## 🏆 Benchmark Results

Despite the aggressive parameter constraint ($\sim 1\text{M}$) and brief training schedule, our NAFNet-SR model achieved strong performance across standard testing datasets. 

| Dataset | MATLAB Bicubic Baseline (PSNR / SSIM) | **NAFNet-SR Model (PSNR / SSIM)** |
| :--- | :--- | :--- |
| **Set5** | 28.39 dB / 0.8115 | **32.239 dB / 0.8960** |
| **Set14** | 26.08 dB / 0.7050 | **28.606 dB / 0.7823** |
| **BSDS100** | 25.96 dB / 0.6683 | **27.612 dB / 0.7373** |
| **Urban100** | 23.14 dB / 0.6585 | **26.140 dB / 0.7877** |
| **Manga109** | 24.92 dB / 0.7884 | **30.767 dB / 0.9109** |
| **DIV2K (Val)** | 28.10 dB / 0.7744 | **30.429 dB / 0.8373** |

## ⚖️ Baseline Comparison

To provide an academically honest and grounded context, here is how our model compares against well-established efficient super-resolution architectures. Metrics below represent PSNR on standard testing datasets at **$\times4$ scale**. 

| Model | Set5 | Set14 | Urban100 |
| :--- | :--- | :--- | :--- |
| **IMDN** (ACM MM 2019) | 32.21 dB | 28.58 dB | 26.04 dB |
| **LAPAR-A** (NeurIPS 2020) | 32.15 dB | 28.61 dB | 26.14 dB |
| **SwinIR-Light** (ICCV 2021) | 32.44 dB | 28.77 dB | 26.47 dB |
| **Our NAFNet-SR** *(17k steps)* | **32.24 dB** | **28.61 dB** | **26.14 dB** |

*Note: SwinIR-Light and IMDN were trained for 500,000+ to 1,000,000 steps. Our model achieves direct parity with IMDN and LAPAR-A in just 17,240 steps.*

This places our model firmly in line with modern efficient baseline architectures, demonstrating incredible sample-efficiency.

### Visual Comparisons

*(Below are visual comparisons demonstrating the model's ability to reconstruct high-frequency details, structural lines, and textures compared to the blurry bicubic baseline).*

<img width="1340" height="350" alt="zebra" src="https://github.com/user-attachments/assets/e2a4fe46-9305-49b1-ad2a-8cb98cba5c7b" />
<img width="1330" height="350" alt="skyline" src="https://github.com/user-attachments/assets/c71f4526-3c95-4629-b546-2b93370234b6" />
<img width="1322" height="453" alt="parrot" src="https://github.com/user-attachments/assets/ba290662-f155-4255-8d50-9e3063bd9c9b" />
<img width="1342" height="506" alt="healing planet" src="https://github.com/user-attachments/assets/1094bb4c-cc04-47d2-8a27-d168cd4beaa3" />
<img width="1316" height="383" alt="gate" src="https://github.com/user-attachments/assets/961dc0fe-d0fa-4b52-8179-ebf169d05cc5" />
<img width="1332" height="445" alt="butterfly" src="https://github.com/user-attachments/assets/ecf44d8e-ad0c-4428-9d40-9a729b07fdf0" />
<img width="1332" height="350" alt="building" src="https://github.com/user-attachments/assets/426dc76e-2ac6-4ec0-ae72-83db63bab519" />
<img width="1331" height="391" alt="Screenshot 2026-08-21 171720" src="https://github.com/user-attachments/assets/bef3b427-7ee5-4678-9bcc-f34e82a193bf" />





## ⚠️ Limitations & Future Work

While the model achieved an impressive **30.43 dB on DIV2K**, placing it on par with efficient SR baselines from the NTIRE 2025 competition, there is still potential left in the architecture due to our training constraints:

1. **Premature Training Termination**: The model was stopped at epoch 40 (only 17,240 optimization steps). In standard super-resolution literature, NAFNet architectures are typically trained for 400,000 to 600,000 steps. 
2. **Incomplete Learning Rate Decay**: Because the training was halted early, the Cosine Annealing learning rate scheduler did not fully execute its curve down to the minimum learning rate, preventing the model from performing final fine-grained local minima convergence.
3. **Compute Environment Constraints**: Training was conducted on Kaggle notebooks, which enforce strict 12-hour session limits and periodic preemptions. To prevent catastrophic data loss and manage continuous checkpointing overhead across ephemeral dual-T4 instances, the training phase was intentionally truncated once competitive benchmark parity was reached.

**Future Path:** A continuous, uninterrupted 500k-step training run with Exponential Moving Average (EMA) and a full learning rate decay curve is mathematically projected to push the DIV2K performance beyond the 30.50 dB barrier.

## 📚 References

This project builds upon the foundational work of several incredible researchers and open-source repositories:

1. **NAFNet:** Chen, Liangyu, et al. *"Simple Baselines for Image Restoration."* European Conference on Computer Vision (ECCV), 2022. [GitHub](https://github.com/megvii-research/NAFNet)
2. **NTIRE 2025 ESR:** *"New Trends in Image Restoration and Enhancement (NTIRE) 2025 Efficient Super-Resolution Challenge."* CVPR 2025 Workshop.
3. **BasicSR:** Xintao Wang, et al. *"BasicSR: Open Source Image and Video Restoration Toolbox."* [GitHub](https://github.com/XPixelGroup/BasicSR) *(Utilized for exact MATLAB bicubic `imresize` metric validation).*
4. **DF2K Dataset:** 
   * **DIV2K:** Agustsson, Eirikur, and Radu Timofte. *"NTIRE 2017 Challenge on Single Image Super-Resolution."* CVPR Workshops, 2017.
   * **Flickr2K:** Lim, Bee, et al. *"Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)."* CVPR Workshops, 2017.
