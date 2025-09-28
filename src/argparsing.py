import sys
from typing import List, Optional
from pydantic import BaseModel
import tyro

from models.model_registry import models

class MinimalArgs(BaseModel):
    model: str

def get_model_and_config():
    def _get_model_args() -> List[str]:
        argv = sys.argv[1:]
        for i in range(len(argv)):
            if argv[i].startswith("--model="):
                return argv[i].split("=")
            elif argv[i].startswith("--model") and i+1 < len(argv):
                return ["--model", argv[i+1]]
        raise ValueError("Missing required --model argument.")
    
    base_args = tyro.cli(MinimalArgs, args = _get_model_args())
    model_cls = models[base_args.model]
    
    config_cls = model_cls.get_args_model()
    config_args = tyro.cli(config_cls)
    
    return model_cls, config_args
    
    
    
    
        
            