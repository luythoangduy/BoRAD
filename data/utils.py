from torchvision import transforms
import numpy as np
from . import TRANSFORMS


def get_transforms(cfg, train, cfg_transforms):
    if cfg_transforms is None:
        return None
    transform_list = []
    for t in cfg_transforms:
        t = {k: v for k, v in t.items()}
        t_type = t.pop('type')

        # Recursively instantiate nested transforms (e.g. for RandomApply)
        for k, v in t.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and 'type' in v[0]:
                nested_list = []
                for nt in v:
                    nt_copy = {nk: nv for nk, nv in nt.items()}
                    nt_type = nt_copy.pop('type')
                    nested_list.append(TRANSFORMS.get_module(nt_type)(**nt_copy))
                t[k] = nested_list

        t_tran = TRANSFORMS.get_module(t_type)(**t)
        transform_list.extend(t_tran) if isinstance(t_tran, list) else transform_list.append(t_tran)
    transform_out = TRANSFORMS.get_module('Compose')(transform_list)

	# if train:
	# 	if cfg.size <= 32:
	# 		transform_out[0] = transforms.RandomCrop(cfg.size, padding=4)
	return transform_out


def make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v
