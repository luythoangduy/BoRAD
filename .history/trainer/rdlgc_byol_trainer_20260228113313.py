"""
Trainer for RDLGC with BYOL-style architecture.

Key differences from original trainer:
1. Calls update_momentum_encoder() after each optimization step
2. Sets total_steps for momentum scheduling
3. Uses BYOL loss (no negative samples)
"""

import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp, read_data, save_data
from util.bank import MemoryBank
import torch.nn.functional as F


@TRAINER.register_module
class RDLGCBYOLTrainer(BaseTrainer):
    """
    Trainer for RDLGC with BYOL-style architecture.
    
    Key features:
    1. Updates momentum encoder after each optimization step
    2. Uses BYOL loss without negative samples
    3. Supports momentum scheduling (constant, cosine, linear)
    """
    
    def __init__(self, cfg):
        super(RDLGCBYOLTrainer, self).__init__(cfg)

        # Handle both DDP and non-DDP cases
        net_module = self.net.module if hasattr(self.net, 'module') else self.net

        # === Setup optimizers ===
        # Create a module list containing proj_layer + predictor for optimizer
        # proj_and_pred = torch.nn.ModuleList([net_module.proj_layer, net_module.predictor])
        # self.optim.proj_opt = get_optim(cfg.optim.proj_opt.kwargs, proj_and_pred, lr=cfg.optim.lr)

        # # Temporarily remove proj_layer and predictor for distill_opt
        # proj_layer = net_module.proj_layer
        # predictor = net_module.predictor
        # net_module.proj_layer = None
        # net_module.predictor = None
        self.optim.distill_opt = get_optim(cfg.optim.distill_opt.kwargs, self.net, lr=cfg.optim.lr * 5)
        # net_module.proj_layer = proj_layer
        # net_module.predictor = predictor

        # === Set total steps for momentum scheduling ===
        total_steps = cfg.trainer.iter_full if hasattr(cfg.trainer, 'iter_full') else \
                     cfg.trainer.epoch_full * cfg.data.train_size
        net_module.set_total_steps(total_steps)

        # === Initialize WandB ===
        self.use_wandb = False
        if hasattr(cfg, 'wandb') and cfg.wandb.enabled and self.master:
            if not WANDB_AVAILABLE:
                log_msg(self.logger, "[WandB] Warning: wandb not installed, logging disabled")
            else:
                # Set API key if provided
                if cfg.wandb.api_key is not None:
                    import os
                    os.environ['WANDB_API_KEY'] = cfg.wandb.api_key

                # Initialize wandb
                run_name = cfg.wandb.name if cfg.wandb.name else f"{cfg.model.name}_{cfg.data.root.split('/')[-1]}"
                wandb.init(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    name=run_name,
                    tags=cfg.wandb.tags,
                    notes=cfg.wandb.notes,
                    config={
                        'model': cfg.model.name,
                        'dataset': cfg.data.root,
                        'batch_size': cfg.trainer.data.batch_size,
                        'lr': cfg.optim.lr,
                        'epochs': cfg.trainer.epoch_full,
                        'momentum_schedule': net_module.momentum_schedule,
                    }
                )
                self.use_wandb = True
                log_msg(self.logger, f"[WandB] Initialized: {wandb.run.name}")

        # Log BYOL-specific info
        if self.master:
            log_msg(self.logger, f"[BYOL] Total steps: {total_steps}")
            log_msg(self.logger, f"[BYOL] Momentum schedule: {net_module.momentum_schedule}")
            if net_module.momentum_schedule != 'constant':
                log_msg(self.logger, f"[BYOL] Momentum range: {net_module.momentum_start} → {net_module.momentum_end}")

    def set_input(self, inputs):
        """Set input data"""
        self.imgs = inputs['img'].cuda()
        self.aug_imgs = inputs.get('aug_img', None)
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.labels = inputs['label'].cuda()
        self.bs = self.imgs.shape[0]
        
        if self.aug_imgs is not None:
            self.aug_imgs = self.aug_imgs.cuda()

    def forward(self):
        """Forward pass"""
        outputs = self.net(self.imgs, self.aug_imgs)
        (self.feats_t, self.feats_s, self.feats_t_k, 
         self.feats_t_q_grid, self.feats_t_k_grid, 
         self.glb_feats, self.glb_feats_k) = outputs

    def backward_term(self, loss_term, optim):
        """Backward pass with gradient clipping"""
        # optim.proj_opt.zero_grad()
        optim.distill_opt.zero_grad()
        
        if self.loss_scaler:
            self.loss_scaler(
                loss_term, optim, 
                clip_grad=self.cfg.loss.clip_grad, 
                parameters=self.net.parameters(),
                create_graph=self.cfg.loss.create_graph
            )
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            
            # optim.proj_opt.step()
            optim.distill_opt.step()

    def optimize_parameters(self):
        """Optimization step with BYOL-style momentum update"""
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        
        with self.amp_autocast():
            self.forward()
            
            # === Reconstruction loss (cosine similarity) ===
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)

            # === Prototype InfoNCE loss ===
            loss_glb = self.loss_terms['proto'](self.glb_feats, self.glb_feats_k, self.labels)
            
            # === BYOL Dense loss (no negatives!) ===
            # Note: q_grid has predictor output, k_grid doesn't
            loss_den = self.loss_terms['dense'](
                self.feats_t,           # q_b: online backbone for matching
                self.feats_t_k,         # k_b: target backbone for matching  
                self.feats_t_q_grid,    # q_grid: online output (with predictor)
                self.feats_t_k_grid,    # k_grid: target output (no predictor)
                self.labels
            )
            
            loss = loss_cos + loss_glb + loss_den

        self.backward_term(loss, self.optim)

        # === CRITICAL: Update momentum encoder after each step ===
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        if hasattr(net_module, 'update_momentum_encoder'):
            net_module.update_momentum_encoder()

        # === Logging ===
        loss_cos_val = reduce_tensor(loss_cos, self.world_size).clone().detach().item()
        loss_glb_val = reduce_tensor(loss_glb, self.world_size).clone().detach().item()
        loss_den_val = reduce_tensor(loss_den, self.world_size).clone().detach().item()
        loss_total_val = reduce_tensor(loss, self.world_size).clone().detach().item()

        update_log_term(self.log_terms.get('cos'), loss_cos_val, 1, self.master)
        update_log_term(self.log_terms.get('proto'), loss_glb_val, 1, self.master)
        update_log_term(self.log_terms.get('dense'), loss_den_val, 1, self.master)

        # WandB logging
        if self.use_wandb and self.iter % self.cfg.wandb.log_interval == 0:
            log_dict = {
                'train/loss_total': loss_total_val,
                'train/loss_cos': loss_cos_val,
                'train/loss_proto': loss_glb_val,
                'train/loss_dense': loss_den_val,
                'train/lr': self.optim.proj_opt.param_groups[0]['lr'],
                'train/epoch': self.epoch,
                'train/iter': self.iter,
            }

            # Log momentum
            current_momentum = net_module.get_current_momentum()
            if isinstance(current_momentum, torch.Tensor):
                current_momentum = current_momentum.item()
            log_dict['train/momentum'] = current_momentum


            wandb.log(log_dict, step=self.iter)

        # Log momentum value periodically
        if self.iter % 100 == 0 and self.master:
            current_momentum = net_module.get_current_momentum()
            if isinstance(current_momentum, torch.Tensor):
                current_momentum = current_momentum.item()
            log_msg(self.logger, f"[BYOL] Step {self.iter}, Momentum: {current_momentum:.4f}")

    @torch.no_grad()
    def test(self):
        """Test/evaluation loop"""
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)

        self.reset(isTrain=False)
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)

        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()

            # Compute cosine loss for logging
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            update_log_term(
                self.log_terms.get('cos'),
                reduce_tensor(loss_cos, self.world_size).clone().detach().item(),
                1, self.master
            )

            # Compute anomaly map using evaluator
            anomaly_map, _ = self.evaluator.cal_anomaly_map(
                self.feats_t, self.feats_s,
                [self.imgs.shape[2], self.imgs.shape[3]],
                uni_am=False,
                amap_mode='add',
                gaussian_sigma=4
            )

            # Binarize ground truth mask
            self.imgs_mask[self.imgs_mask > 0.5] = 1
            self.imgs_mask[self.imgs_mask <= 0.5] = 0

            # Visualization if enabled
            if self.cfg.vis:
                from util.vis import vis_rgb_gt_amp
                root_out = self.cfg.vis_dir if self.cfg.vis_dir is not None else self.writer.logdir
                vis_rgb_gt_amp(
                    self.img_path, self.imgs,
                    self.imgs_mask.cpu().numpy().astype(int),
                    anomaly_map,
                    self.cfg.model.name, root_out,
                    self.cfg.data.root.split('/')[1]
                )

            # Save results to disk
            save_path = os.path.join(self.cfg.logdir, 'results')
            save_data(
                save_path, self.cls_name, self.img_path,
                self.imgs_mask.cpu().numpy().astype(int),
                anomaly_map,
                self.anomaly.cpu().numpy().astype(int)
            )

            t2 = get_timepc()
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None

            # Logging
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(
                        self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'),
                        self.master, None
                    )
                    log_msg(self.logger, msg)

        # Compute metrics per class
        if self.master:
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                imgs_masks, anomaly_maps, anomalys, cls_names_data = read_data(save_path, cls_name)
                results = dict(
                    imgs_masks=imgs_masks,
                    anomaly_maps=anomaly_maps,
                    anomalys=anomalys,
                    cls_names=cls_names_data
                )

                results = {k: np.array(v) for k, v in results.items()}
                metric_results = self.evaluator.run(results, cls_name, self.logger)

                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None

                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1

                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

                    if avg_act:
                        # Compute average across all classes
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

            # Print table
            msg = tabulate.tabulate(msg, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center', stralign='center')
            log_msg(self.logger, '\n' + msg)

            # Log metrics to WandB
            if self.use_wandb:
                wandb_metrics = {}
                for metric in self.metrics:
                    # Log individual class metrics
                    for cls_name in self.cls_names:
                        metric_val = self.metric_recorder[f'{metric}_{cls_name}'][-1]
                        wandb_metrics[f'test/{metric}_{cls_name}'] = metric_val

                    # Log average metric
                    if f'{metric}_Avg' in self.metric_recorder and len(self.metric_recorder[f'{metric}_Avg']) > 0:
                        avg_val = self.metric_recorder[f'{metric}_Avg'][-1]
                        wandb_metrics[f'test/{metric}_avg'] = avg_val

                wandb_metrics['test/epoch'] = self.epoch
                wandb.log(wandb_metrics, step=self.iter)

        return None


# ============================================================================
# Alternative: Minimal changes to existing trainer
# ============================================================================

def patch_existing_trainer(trainer_class):
    """
    Decorator to patch existing trainer with BYOL momentum updates.
    
    Usage:
        @patch_existing_trainer
        class RDLGCTrainer(BaseTrainer):
            ...
    """
    original_init = trainer_class.__init__
    original_optimize = trainer_class.optimize_parameters
    
    def new_init(self, cfg):
        original_init(self, cfg)
        
        # Add predictor to optimizer
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        if hasattr(net_module, 'predictor'):
            # Add predictor params to proj_opt
            predictor_params = list(net_module.predictor.parameters())
            for param in predictor_params:
                self.optim.proj_opt.add_param_group({'params': param})
            
            # Set total steps
            total_steps = cfg.trainer.iter_full if hasattr(cfg.trainer, 'iter_full') else \
                         cfg.trainer.epoch_full * cfg.data.train_size
            net_module.set_total_steps(total_steps)
    
    def new_optimize(self):
        original_optimize(self)
        
        # Update momentum encoder
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        if hasattr(net_module, 'update_momentum_encoder'):
            net_module.update_momentum_encoder()
    
    trainer_class.__init__ = new_init
    trainer_class.optimize_parameters = new_optimize
    
    return trainer_class