from __future__ import annotations

import asteroid
import torch


def build_pit_l1_loss():
    base_loss = torch.nn.L1Loss()
    return asteroid.losses.pit_wrapper.PITLossWrapper(base_loss, pit_from="pw_pt")

