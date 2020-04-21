# [Boundary-Aware Feature Propagation for Scene Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Boundary-Aware_Feature_Propagation_for_Scene_Segmentation_ICCV_2019_paper.pdf)

1. Install pytorch 

  - The code is implemented on python3.6 and official [Pytorch](https://github.com/pytorch/pytorch/tree/fd25a2a86c6afa93c7062781d013ad5f41e0504b#from-source).
  - Please install [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) for Synchronized BN. 

2. Usage

  - Training:
  
   ```shell
   python train.py --model  bfp --dataset pascalcontext --checkname bfp101  --batch-size 12
   ```
   
  - Testing:
  
   ```shell
   python test.py --model bfp --dataset pascalcontext --resume-dir model_path --eval
   ```  
3. Citations

  - Please consider citing our paper in your publications if the project helps your research.
```
@inproceedings{ding2019bfp,
  title={Boundary-aware feature propagation for scene segmentation},
  author={Ding, Henghui and Jiang, Xudong and Liu, Ai Qun and Thalmann, Nadia Magnenat and Wang, Gang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={6819--6829},
  year={2019}
}
```

