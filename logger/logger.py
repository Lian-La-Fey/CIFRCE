import os
import logging

os.environ["DEEPSPEED_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

def get_logger(name: str = "app") -> logging.Logger:
    """
    Logger called by the same name is always the same object. (singleton)
    However, if handlers are added repeatedly, the logs are printed two or three times.
    Therefore:
        configure the logger once
        in subsequent calls, return the same logger
        !controlled singleton!
    """
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger