import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
warnings.filterwarnings("ignore")
import setproctitle
import torch
import numpy as np
import random
setproctitle.setproctitle("Duy is training")

def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('-c', '--cfg_path', default='configs/rd_mvtec_debug.py')
	# parser.add_argument('-c', '--cfg_path', default='configs/invad_mvtec_debug.py')
	parser.add_argument('-c', '--cfg_path', default='configs/rd/rd_mvtec.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--data_path', type=str, default=None, help='Path to dataset root directory')
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER,)
	cfg_terminal = parser.parse_args()
	cfg = get_cfg(cfg_terminal)
	run_pre(cfg)
	init_training(cfg)
	init_checkpoint(cfg)
	trainer = get_trainer(cfg)
	trainer.run()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
	setup_seed(42)
	main()
