# A Joint Modeling of Vision-Language-Action for Target-oriented Grasping in Clutter
This is the official repository for the paper: A Joint Modeling of **Vi**sion-**L**anguage-Action for Target-oriented **G**rasping in Clutter (ICRA 2023).

Paper: https://arxiv.org/abs/2302.12610

<video width="1280" height="720" controls>
  <source src="videos/vilg_v4_20Mb.mp4" type="video/mp4">
</video>

We focus on the task of language-conditioned grasping in clutter, in which a robot is supposed to grasp the target object based on a language instruction. Previous works separately conduct visual grounding to localize the target object, and generate a grasp for that object. However, these works require object labels or visual attributes for grounding, which calls for handcrafted rules in planner and restricts the range of language instructions. In this paper, we propose to jointly model vision, language and action with object-centric representation. Our method is applicable under more flexible language instructions, and not limited by visual grounding error. Besides, by utilizing the powerful priors from the pre-trained multi-modal model and grasp model, sample efficiency is effectively improved and the sim2real problem is relived without additional data for transfer. A series of experiments carried out in simulation and real world indicate that our method can achieve better task success rate by less times of motion under more flexible language instructions. Moreover, our method is capable of generalizing better to scenarios with unseen objects and language instructions.

![system overview](images/system.png)

#### Contact

Any question, please let me know: kcxu@zju.edu.cn

## Setup
###  Installation

- Ubuntu 18.04
- Torch==1.7.1, Torchvision==0.8.2
- Pybullet (simulation environment)
- Cuda 11.1
- GTX 3060, 12GB memory is tested

```
git clone git@github.com:xukechun/Vision-Language-Grasping.git
cd ViLG

conda create -n vilg python=3.8
conda activate vilg

pip install -r requirements.txt

python setup.py develop

cd models/graspnet/pointnet2
python setup.py install

cd ../knn
python setup.py install
```
### Assets
We provide the processed object models in this link (TODO). Please download the file and unzip it in the `assets` folder.

## Training

```
python train.py
```

## Evaluation
To test the pre-trained model, simply change the location of `--model_path`:

```python
python test.py --load_model True --model_path 'PATH OF YOUR CHECKPOINT FILE'
```

## Citation

If you find this work useful, please consider citing:

```
@INPROCEEDINGS{10161041,
  author={Xu, Kechun and Zhao, Shuqi and Zhou, Zhongxiang and Li, Zizhang and Pi, Huaijin and Zhu, Yifeng and Wang, Yue and Xiong, Rong},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={A Joint Modeling of Vision-Language-Action for Target-oriented Grasping in Clutter}, 
  year={2023},
  pages={11597-11604},
  doi={10.1109/ICRA48891.2023.10161041}}
```

