# BorAD: Bootstrap Your Own Representations for Anomaly Detection

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Download datasets to `data/` folder:
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- [VisA](https://github.com/amazon-science/spot-diff)
- [BTAD](https://github.com/pankajmishra000/VT-ADL)
- [Real-IAD](https://realiad4ad.github.io/Real-IAD/)

## Training

```bash
# MVTec
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/rd/rd_byol_mvtec.py -m train

# VisA
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/rd/rd_byol_visa.py -m train

# BTAD
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/rd/rd_byol_btad.py -m train

# Real-IAD
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/rd/rd_byol_realiad.py -m train
```

### Multi-GPU (DDP)

```bash
./scripts/run_ddp.sh configs/rd/rd_byol_mvtec.py 8
```

### Run All Experiments

```bash
# Sequential (single GPU)
./scripts/run_experiments.sh

# Parallel (multi-GPU)
./scripts/run_experiments_parallel.sh 4
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/rd/rd_byol_mvtec.py -m test
```

## Citation

```bibtex
@article{borad2025,
  title={BorAD: Bootstrap Your Own Representations for Anomaly Detection},
  author={},
  journal={},
  year={2025}
}
```

## Acknowledgement

Built on [ADer](https://github.com/zhangzjn/ADer), [RD4AD](https://github.com/hq-deng/RD4AD), and [AD-LGC](https://github.com/LGC-AD/AD-LGC.git).
