# Project Name

GaitMotion: A Multitask Dataset for Pathological Gait Forecasting

![Project Screenshot](./figure/figure0.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Implementation](#implementation)
- [Citation](#citation)
- [Contributing](#contributing)
<!-- - [License](#license) -->

## Introduction

We provide a dataset for the gait analysis which contains extensive ground truths from gait events at the sample index level to gait parameters for normal and pathological walking patterns. We also provide a baseline model to predict the gait parameters from IMU recordings. 

![Project Screenshot](figure/figure0.png)

## Features

This paper presents a comprehensive dataset on gait and an architecture for analyzing it. Gait deviations are closely related to specific parameters in the gait cycle, containing a combination of stance and swing phases with important events such as heel-strike and toe-off. Our dataset includes parameters on three different tasks that are crucial for healthcare professionals to assess disease and evaluate risk. The gait cycle consists of multiple critical states that support normal walking. Any failure in gait events can lead to the risk of falling. The on-off-ground status provides precise timing for heel-strike and toe-off events. The stance and swing phases, including details on single and double support times, determine the stability and balance during locomotion. Gait deviation is a significant indicator of disease conditions and is commonly associated with joint pathology, decreased muscular strength, range of motion constraints, and more. GaitMotion has rich ground truth labels which could support stride-to-stride fluctuation analysis in different types of walking. 

![Project Screenshot](figure/figure1.png)

## Implementation

To run the training:

```
python train.py
```

Change the parameter accordingly. In the Common_fun.py, the seq_length parameter controls the step segmentation length. The subID is the participant ID that you hope to test with. The model will train the remaining subjects and test the subID participant. 

## Citation

If you find the dataset or code useful, please cite our papers:

{}

## Contributing

We express our gratitude to the volunteers who participated in the data collection experiment. Thanks for Prof. Calvin Kuo's guidance on the data collection and arrangement. 

<!-- ## License

Specify the license under which your project is distributed. For example:

This project is licensed under the [MIT License](LICENSE). -->
