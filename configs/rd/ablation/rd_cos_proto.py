"""Ablation: CosLoss + PrototypeInfoNCELoss (no dense)"""
from configs.rd.rd_byol_mvtec import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()

        self.loss.loss_terms = [
            dict(type='CosLoss', name='cos', avg=False, lam=1.0),
            dict(type='PrototypeInfoNCELoss', name='proto', lam=1.0, n_prototypes=5, temperature=0.07),
        ]

        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>7.6f'),
            dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
            dict(name='proto', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
