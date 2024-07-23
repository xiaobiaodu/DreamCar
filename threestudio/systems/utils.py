import sys
import warnings
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import threestudio


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model, pose_model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        threestudio.debug(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()

    params_exclude_posemlp = []
    params_lambda_view = []
    for name, param in model.named_parameters():
        if "posemlp" not in name and "lambda_view" not in name:

            params_exclude_posemlp.append(param)  # These parameters will be updated
        if "lambda_view"  in name:
            params_lambda_view.append(param)

    params = [
        {"params": params_exclude_posemlp, **config.args},
        {"params": pose_model.parameters(), "lr":config.pose_lr},
        {"params": params_lambda_view, "lr":config.view_lr}
    ]
    optim = getattr(torch.optim, config.name)(params)

    return optim


def parse_scheduler_to_instance(config, optimizer):
    if config.name == "ChainedScheduler":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.ChainedScheduler(schedulers)
    elif config.name == "Sequential":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=config.milestones
        )
    else:
        scheduler = getattr(lr_scheduler, config.name)(optimizer, **config.args)
    return scheduler


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler
