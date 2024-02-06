# Lower Ricci Curvature for Efficient Community Detection
## Overview
This is a repository for a paper "Lower Ricci Curvature for Efficient Community Detection" published in 2024. 

This study introduces the Lower Ricci Curvature (LRC), a novel, scalable, and scale-free discrete curvature designed to enhance community detection in networks. Addressing the computational challenges posed by existing curvature-based methods, LRC offers a streamlined approach with linear computational complexity, making it well-suited for large-scale network analysis. We further develop an LRC-based preprocessing method that effectively augments popular community detection algorithms. Through comprehensive simulations and applications on real- world datasets, including the NCAA football league network, the DBLP collaboration network, the Amazon product co-purchasing network, and the YouTube social network, we demonstrate the efficacy of our method in significantly improving the performance of various community detection algorithms.
## Package Requirement
- Python >= 3.9
- [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature/tree/v0.5.1)
- [CDlib](https://github.com/GiulioRossetti/cdlib/tree/master)
- [PyTorch](https://pytorch.org/get-started/locally/) >= 2.1.2
- [Networkx](https://networkx.org/documentation/stable/install.html)
- [matplotlib](https://matplotlib.org) >= 3.8.2
- [numpy](https://numpy.org) >= 1.26.2
- [pandas](https://pandas.pydata.org) >= 2.1.4
- [seaborn](https://seaborn.pydata.org) >= 0.13.0
- torchVision
- torchaudio
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 11.8
## Reference
- Ye, Z., Liu, K. S., Ma, T., Gao, J., & Chen, C. (2019, September). Curvature graph network. In International conference on learning representations.
- Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., & Bronstein, M. M. (2021). Understanding over-squashing and bottlenecks on graphs via curvature. arXiv preprint arXiv:2111.14522.
- Rossetti, G., Milli, L., & Cazabet, R. (2019). CDLIB: a python library to extract, compare and evaluate communities from complex networks. Applied Network Science, 4(1), 1-26.
