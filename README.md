# PointNet++ for 3D Point Cloud Classification

## Overview

This repository contains an implementation of PointNet++, a deep learning architecture designed for 3D point cloud classification. The notebook performs Exploratory Data Analysis (EDA) on 3D object data in .OFF format, extracts vertices and faces, and processes them for neural network training.

The project uses the ModelNet40 dataset, a widely used benchmark dataset for 3D object recognition.

### Features
*Data Exploration*:

* Reads .OFF files to extract vertices and faces.
* Maps dataset categories into a dictionary (hashmap).
* Visualizes 3D point clouds using Plotly.


*Preprocessing*:

* Implements data loaders for ModelNet40.
* Normalizes and converts 3D point data into a structured format.


*PointNet++ Model*:

* Defines the PointNet++ architecture for classification.
* Trains the model on ModelNet40 data.


*Evaluation & Visualization*:

* Computes accuracy and loss during training.
* Provides 3D visualization of input point clouds.


## Usage

1. Prepare the dataset:
   * Ensure ModelNet40 dataset is available at: <ModelNet40-dataset>
