# A Dual-Modal Mixture-of-Experts Attention U-Net (DMoE-AttU-Net) for Change Detection Using Heterogeneous Optical and SAR Remote Sensing Images

Ehsan Khankeshizadeh, Ali Mohammadzadeh, [Ali Jamali](https://www.researchgate.net/profile/Ali-Jamali), and Sadegh Jamali


Citation
---------------------

**Please kindly cite the paper if this code is useful and helpful for your research.**

      @article{Ehsan2026,
        title = {A Dual-Modal Mixture-of-Experts Attention U-Net (DMoE-AttU-Net) for Change Detection Using Heterogeneous Optical and SAR Remote Sensing Images},
        author = {Khankeshizadeh, Seyed Ehsan and Mohammadzadeh, Ali and Jamali, Ali and Jamali, Sadegh },
        journal = {Remote Sensing},
        volume = {18},
        pages = {},
        year = {2026},
        issn = {2072-4292},
        doi = {https://doi.org/10.3390/rs18101508},
        url = {https://www.mdpi.com/2072-4292/18/10/1508}
      }


<img src="Architecture.png"/>

Figure 1. (a) Overall architecture of the proposed Dual-Modal Mixture-of-Experts Attention U-Net (DMoE-AttU-Net). The SE attention module integrated within each CNN expert is detailed in Figure 2. 𝜓1 and 𝜓1 denote attention maps used for feature modulation in the model.


<img src="arch2.jpg"/>

Figure 2. Schematic of the squeeze-and-excitation (SE) channel attention module used within each CNN expert in the SAR branch of the proposed architecture (see Figure 1).

<img src="results.jpg"/>

Figure 3. Visual comparisons of the change maps obtained by different DL models on the three datasets. green is used to represent true positives (TP), while black represents true negatives (TN). Red signifies false positives (FP) and blue indicates false negatives (FN). The dotted-line boxes highlight selected regions for detailed comparison, where the proposed DMoE-AttU-Net demonstrates lower error rates and better preservation of change regions.


## License

Copyright (c) 2026 Ali Jamali. Released under the MIT License. See [LICENSE](LICENSE) for details.

