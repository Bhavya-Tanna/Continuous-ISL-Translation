## Continuous Indian Sign Language Translation Using Deep Learning

A deep learning project focused on translating continuous Indian Sign Language (ISL) video sequences into meaningful English text.

## Project Overview

Communication between the Deaf and hearing communities can be challenging when a common language is not available. This project aims to develop an AI-based system capable of understanding continuous Indian Sign Language and generating corresponding English sentences.

Unlike isolated sign recognition systems that classify individual gestures, this project focuses on **continuous sign language translation**, where temporal context across a sequence of signs is used to generate meaningful English text.

## Objective

The primary objective is to build a deep learning pipeline that:

- Accepts ISL video sequences as input.
- Extracts visual and spatial information from signing.
- Captures temporal relationships between signs.
- Uses multimodal information such as RGB features and hand/body landmarks.
- Generates an English sentence corresponding to the signed sequence.
- Can eventually operate on real-time webcam input.

## Planned Architecture

```text
Webcam / Video
       ↓
Frame Sampling
       ↓
MediaPipe + Visual Feature Extraction
       ↓
Multimodal Feature Fusion
       ↓
Temporal Transformer
       ↓
English Sentence
       ↓
Optional Text-to-Speech
```

## Dataset

The primary dataset used in this project is iSign, a large-scale Indian Sign Language dataset.

A selected subset of the dataset is being used for experimentation due to computational and storage constraints.

The dataset preparation pipeline includes:

Metadata inspection and profiling
Sentence quality filtering
Dataset subset selection
ZIP archive indexing
Selective video retrieval
Video integrity validation
Dataset characteristic analysis

The dataset itself is not included in this repository. Users must obtain and use the dataset according to its original license and terms.


## Current Development Pipeline

Dataset Metadata
      ↓
Metadata Profiling
      ↓
Dataset Selection
      ↓
ZIP Archive Indexing
      ↓
Selective Video Retrieval
      ↓
Video Integrity Validation
      ↓
Dataset Characterization
      ↓
Preprocessing
      ↓
Model Training
      ↓
Evaluation
      ↓
Real-Time Deployment


## Repository Structure

continuous-isl-translation/
│
├── data/
│   ├── raw/
│   ├── metadata/
│   └── processed/
│
├── src/
│   ├── inspect_metadata.py
│   ├── profile_metadata.py
│   ├── select_dataset.py
│   ├── build_video_index.py
│   ├── download_selected_videos.py
│   └── ...
│
├── notebooks/
├── models/
├── outputs/
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md


## Technology Stack

Python
PyTorch / TensorFlow
OpenCV
MediaPipe
NumPy
Pandas
Hugging Face Hub
Scikit-learn
Transformer-based deep learning models

## Note:
License and Dataset Usage

This repository contains project code and documentation. The original dataset remains subject to its own license and usage restrictions.

Do not redistribute the original dataset or downloaded dataset files through this repository.