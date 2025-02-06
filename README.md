# PointNet++ for 3D Point Cloud Classification

This repository implements **PointNet++**, a deep learning framework for 3D point cloud classification. The implementation includes **exploratory data analysis**, **data preprocessing**, **augmentation**, and **training of the PointNet++ model** using the **ModelNet40** dataset.

## 🔹 Introduction

**PointNet++** is an extension of **PointNet**, designed to learn local structures in **point clouds** using a **hierarchical feature learning approach**. It improves upon PointNet by introducing **set abstraction layers** that group points based on a distance metric, enabling **multi-scale feature extraction** and handling **non-uniform sampling densities** effectively.

The methodology follows **three major steps**:
1. **Data Preparation & Augmentation** - Converts **mesh data (.OFF files)** into **point clouds**.
2. **Dataset Transformation** - Extracts **features** and applies **preprocessing techniques**.
3. **Model Training & Evaluation** - Implements **PointNet++**, including **hierarchical set abstraction, feature propagation, and adaptive sampling**.

---

## 📌 Exploratory Data Analysis - Object File Format for 3D Shapes

### 1. Identifying the Classes and Mapping to Hashmap
- The **ModelNet40** dataset contains **40 categories** of 3D objects.
- We construct a **hashmap** mapping **category names** to **numerical labels**.
- This is necessary for **label encoding** during training.

### 2. Exploring a Sample Data Point (.OFF File) and Extracting Vertices & Faces
- **OFF (Object File Format)** stores 3D geometry.
- It consists of:
  - **Vertices** (3D coordinates of points)
  - **Faces** (Triangular connections between vertices)
- We parse the **vertices** and **faces** for point cloud generation.

### 3. Displaying the Sample 3D Shape

#### 3.1 3D Mesh Display
- We reconstruct the **mesh** using **faces** and visualize it using **Plotly**.

#### 3.2 3D Scatter Display
- Instead of **connecting faces**, we plot **only the vertices**.
- This helps in **understanding the distribution of 3D points**.

---

## 📌 Data Conversion - Vertices and Faces to Point Cloud Data

### 1. Point Sampler: Converting Mesh to Point Cloud
Since the dataset provides only **vertices and faces**, we need to **sample points from the surface**.

#### Steps:
1. **Triangle Area Calculation**:
   - Compute area using **Heron’s formula**.
   - Larger triangles are **sampled more frequently**.

2. **Random Point Sampling**:
   - Generate **barycentric coordinates** to **randomly sample points** inside triangles.
   - Ensures a **uniform** distribution over the object’s surface.

3. **Weighted Sampling**:
   - Use **random.choices()** to **select triangles** with probability proportional to their area.
   - Generates a **balanced point cloud**.

4. **Store Sampled Points**:
   - The final output is a **NumPy array** of shape `(num_points, 3)`, representing the **(x, y, z) coordinates**.

---

## 📌 Data Augmentation

### 1. Normalize Data
- Ensures the **point cloud** is **centered** at `(0,0,0)`.
- Scales the point cloud to **fit within a unit sphere**.

### 2. Random Rotation Along Z-axis
- Rotates the point cloud to **increase rotational invariance**.

### 3. Random Noise Addition
- Adds **Gaussian noise** to prevent **overfitting**.

### 4. Convert Point Cloud to Tensor
- Converts the **NumPy array** into a **PyTorch Tensor** for model training.

---

## 📌 Dataset Preparation

### 1. Data Transformation Methods
- Applies **normalization, augmentation, and conversion** on-the-fly.

### 2. Data Preprocessing Methods
- Converts **raw OFF files** into **point clouds**.
- Ensures **consistent** sampling and scaling.

### 3. Dataset Definition & Data Loaders
- Implements a **custom PyTorch Dataset Class** for ModelNet40.
- Uses `torch.utils.data.DataLoader()` for **batch processing**.

---

## 📌 Model Definition: Understanding PointNet++

### 1. Architecture Overview
**PointNet++** extends **PointNet** by introducing **hierarchical feature learning**:

- Uses **Set Abstraction Layers** to **downsample** point clouds while capturing local features.
- Implements **Single-Scale Grouping (SSG)**
- Adapts to **non-uniform point densities**.

### 2. Set Abstraction Layers
- The **core component** of **PointNet++**.
- Divides points into **local neighborhoods** using:
  - **Farthest Point Sampling (FPS)** for selecting key points.
  - **Ball Query or kNN** for grouping neighbors.
  - **PointNet** to extract local features.

### 3. Feature Propagation
- **Interpolates missing points** in subsampled layers.
- Uses **skip connections** to retain **fine-grained details**.

### 4. PointNet++ Loss Function
- Uses **Cross-Entropy Loss** for classification.
- Includes **Batch Normalization** and **Dropout** for regularization.

### 5. Training Function
1. **Load Data** via `DataLoader`.
2. **Forward Pass** through **PointNet++**.
3. **Compute Loss** using **cross-entropy**.
4. **Backpropagation** using the **Adam optimizer**.
5. **Track Accuracy** and adjust the **learning rate dynamically**.

---

## 🚀 Getting Started

### 🔹 Installation
My Virtual enviroments name is *Heimdall* (insprired from the character that bridges connection between different realms).
The environment is developed in the ```SOL supercomputer HPC Cluster``` at ASU supercomputing cluster workspace. 
As foundation Heimdall consist of **Pytorch 2.4.1** with **CUDA 11.8** and **cuDNN 9.3.0** direcly forged from Pytorch and Nvidia channel, and is supported over **Python 3.11.0**. 
The installation procedure is explained in detail below for future reference.

```
mamba create -n venv.heimdall -c conda-forge python=3.11.0
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
```

*For 3D mesh visualization*
```
mamba install plotly -c conda-forge
```

*Mime type rendering requires nbformat>=4.2.0 and installed nbformat=5.10.4 along with tqdm*
```
mamba install nbformat -c conda-forge
mamba install tqdm
```


