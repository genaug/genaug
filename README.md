# GenAug

[**GenAug: Retargeting behaviors to unseen situations via Generative Augmentation**](https://arxiv.org/abs/2302.06671)  
[Zoey Chen](https://qiuyuchen14.github.io//), [Sho Kiami](https://www.linkedin.com/in/shokiami), [Abhishek Gupta*](https://abhishekunique.github.io/) [Vikash Kumar*](https://vikashplus.github.io/)  
[RSS 2023](https://roboticsconference.org/) 

GenAug is a data augmentation tool that leverate text-to-image generative models and generate diverse RGBD images for robotics data collection. 
For the latest updates, see: [genaug.github.io](https://genaug.github.io)

![](media/augmented_combined.png)



## TODOs: 
- [ ] push to pip install
- [ ] clean up and push real-world robot code
- [ ] clean up and push sim experiments
- [ ] (if have time) integrate with SAM and do an interative demo on hugging face
## Guides

- Getting Started: [Installation](#installation), [Quick Tutorial](#quickstart), [Checkpoints & Objects](#download), [Hardware Requirements](#hardware-requirements), [Model Card](model-card.md)
- Data Generation: [Dataset](#dataset-generation), [Tasks](cliport/tasks)
- References: [Citations](#citations), [Acknowledgements](#acknowledgements)




## Installation

Clone Repo:
```bash
git clone https://github.com/genaug/genaug.git
```

## Quickstart
```bash
python genaug.py
```


## Citations
**GenAug**
```bibtex
@article{chen2023genaug,
  title={GenAug: Retargeting behaviors to unseen situations via Generative Augmentation},
  author={Chen, Zoey and Kiami, Sho and Gupta, Abhishek and Kumar, Vikash},
  journal={arXiv preprint arXiv:2302.06671},
  year={2023}
}
```

**Stable Diffusion**
```bibtex
@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10684--10695},
  year={2022}
}
```


## Questions or Issues?

Please file an issue with the issue tracker.  