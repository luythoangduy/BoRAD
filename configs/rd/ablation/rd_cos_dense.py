"""Ablation: CosLoss + BYOLDenseLoss (no prototype)"""
from configs.rd.rd_byol_mvtec import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()

        self.loss.loss_terms = [
            dict(type='CosLoss', name='cos', avg=False, lam=1.0),
            dict(type='BYOLDenseLoss', name='dense', lam=1.0, use_spatial_matching=True),
        ]

        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>7.6f'),
            dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='dense', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
