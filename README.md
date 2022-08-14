# CheXRelNet: An Anatomy-Aware Model for Tracking Longitudinal Relationships between Chest X-Rays

This is code repository for the [paper](https://arxiv.org/abs/2208.03873).
<br />
**CheXRelNet: An Anatomy-Aware Model for Tracking Longitudinal Relationships between Chest X-Rays**
by [Gaurang Karwande, Amarachi Mbakawe, Joy T. Wu, Leo A. Celi, Mehdi Moradi, and Ismini Lourentzou]
<br />
<p align='justify'> Despite the progress in utilizing deep learning to automate chest radiograph interpretation and disease diagnosis tasks, change between sequential Chest X-rays (CXRs) has received limited attention. Monitoring the progression of pathologies that are visualized through chest imaging poses several challenges in anatomical motion estimation and image registration, i.e., spatially aligning the two images and modeling temporal dynamics in change detection. In this work, we propose CheXRelNet, a neural model that can track longitudinal pathology change relations between two CXRs. CheXRelNet incorporates local and global visual features, utilizes inter-image and intra-image anatomical information, and learns dependencies between anatomical region attributes, to accurately predict disease change for a pair of CXRs. Experimental results on the Chest ImaGenome dataset show increased downstream performance compared to baselines. </p>

If you find this code, models or results useful, please cite us using the following bibTex:
```
@inproceedings{karwande2022cxr,
  title={{CheXRelNet: An Anatomy-Aware Model for Tracking Longitudinal Relationships between Chest X-Rays}},
  author={Karwande, Gaurang and Mbakwe, Amarachi and Wu, Joy T. and Celi, Leo Antony and Moradi, Mehdi and Lourentzou, Ismini},
  year={2022},
  booktitle={Proceedings of the 25th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)}
  }
```
<figure>
<p align='center'>
<img src='https://github.com/Gaurangkarwande/ChexRelNet/blob/master/figures/model.jpg' width='600'/, cap>
</p>
<figcaption>ChexRelNet</figcaption>
</figure>

### Package Dependencies
- Python version 3.7.6
- matplotlib==3.1.3
- opencv-python==4.5.3.56
- numpy==1.19.5
- pandas==1.0.1
- pip==21.3
- pillow==7.0.0
- pytables==3.6.1
- scikit-learn==0.22.1
- torch==1.9.1
- torch_geometric==2.0.1
- torchvision==0.10.1
