"""Ablation: Full model WITHOUT predictor (CosLoss + BYOLDenseLoss + PrototypeInfoNCELoss)"""
from configs.rd.rd_byol_mvtec import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()

        # Disable predictor for ablation
        self.model.kwargs['use_predictor'] = False
