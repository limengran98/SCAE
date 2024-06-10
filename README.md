# SCAE
The code of the paper SCAE: Structural Contrastive Auto-encoder for Incomplete Multi-view Representation Learning

This repository builds upon the work of [TPAMI2022-DCP](https://github.com/XLearning-SCU/2022-TPAMI-DCP). We extend our gratitude to all the authors of this work. The datasets can be found [here](https://github.com/XLearning-SCU/2022-TPAMI-DCP).

## Environment Dependencies

To ensure smooth execution, please install the following dependencies:

- `pytorch>=1.2.0`
- `numpy>=1.19.1`
- `scikit-learn>=0.23.2`
- `munkres>=1.1.4`
- `PyGCL>=0.1.0`

## Usage

To run the multi-view clustering script, use:
```bash
python run_clustering_multiview.py
```
To run the supervised multi-view script, use:
```bash
python run_supervised_multiview.py
```
To run the human action recognition tasks, use:
```bash
python run_HAR.py
```
## Citation

If you find our work useful, please cite the following articles:

```bibtex
@article{liscae,
  title={SCAE: Structural Contrastive Auto-encoder for Incomplete Multi-view Representation Learning},
  author={Li, Mengran and Zhang, Ronghui and Zhang, Yong and Piao, Xinglin and Zhao, Shiyu and Yin, Baocai},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2024},
  doi={10.1145/3672078}
}

@article{lin2022dual,
  title={Dual contrastive prediction for incomplete multi-view representation learning},
  author={Lin, Yijie and Gou, Yuanbiao and Liu, Xiaotian and Bai, Jinfeng and Lv, Jiancheng and Peng, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={4},
  pages={4447--4461},
  year={2022},
  publisher={IEEE}
}
```
```

Feel free to further modify or expand it as needed.
