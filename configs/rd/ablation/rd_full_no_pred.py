"""Ablation: CosLoss + BYOLDenseLoss + PrototypeInfoNCELoss, WITHOUT predictor"""
from configs.rd.rd_byol_mvtec import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()

        # Disable predictor — full loss but no BYOL asymmetry
        self.model.kwargs['use_predictor'] = False
