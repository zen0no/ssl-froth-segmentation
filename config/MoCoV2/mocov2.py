from mmengine.config import read_base

from mmpretrain.models.selfsup.moco import MoCo
from mmpretrain.models.backbones import ResNet
from mmpretrain.models.necks import MoCoV2Neck
from mmpretrain.models.heads import ContrastiveHead


with read_base():
    from .._base_.dataset import *
    from .._base_.runtime import *

model = dict(
    type=MoCo,
    queue_len=65536,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type=ResNet,
        depth=50,
        norm_cfg=dict(type="BN"),
        zero_init_residual=False,)
    neck=dict(
        type=MoCoV2Neck,
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type=ContrastiveHead
        loss=dict(type="CrossEntropyLoss"),
        temperature=0.2)
)


default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

auto_scale_lr = dict(base_batch_size=256)

