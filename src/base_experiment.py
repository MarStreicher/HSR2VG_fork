from pydantic import BaseModel
from typing import List


class BaseExperimentArgs(BaseModel):
    target_domain: str
    source_domain: str
    use_wandb: bool = False
    sweep: bool = False
