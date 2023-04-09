import torch
import os
import typing
from flytekit import workflow
from project.wf_33_275.main import Hyperparameters
from project.wf_33_275.main import wf

_wf_outputs=typing.NamedTuple("WfOutputs",wf_0=torch.nn.modules.module.Module)
@workflow
def wf_33(_wf_args:Hyperparameters)->_wf_outputs:
	wf_o0_=wf(hp=_wf_args)
	return _wf_outputs(wf_o0_)