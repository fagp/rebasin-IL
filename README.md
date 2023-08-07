# RebasinIL Implementation

This repository contains the official implementation of RebasinIL, a continual learning approach proposed in the paper [Re-basin via implicit Sinkhorn differentiation](https://openaccess.thecvf.com/content/CVPR2023/papers/Pena_Re-Basin_via_Implicit_Sinkhorn_Differentiation_CVPR_2023_paper.pdf), accepted at CVPR 2023.

## Overview
RebasinIL introduces an new perspective to perform incremental learning using the re-basin technique. The provided results are very encouraging, although more experimentation is needed. The repo provides the implementation as an Avalanche plugin. All the metrics, benchmarks, and incremental learning techniques used are implemented on Avalanche.

## Contents
* **Avalanche Plugin:** The RebasinIL plugin for the Avalanche framework can be found on `utils` folder.
* **Incremental Learning scripts:** Call the provided scripts to reproduce the reported results using the RotatedMnist benchmark.
* **`print_metrics.py`:** Use this script to generate a readable output of metrics and results from the Avalanche log.

 
## Getting Started

Install the project dependecies by doing `pip install -r requirements.txt`.

Please note that this codebase utilizes a previous version of the Sinkhorn re-basin network. Compatibility with the latest version is not guaranteed. Adaptations may be needed for integration with the current iteration.
