# server.py
import os
import argparse
import uvicorn
import hydra
import numpy as np
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf
import logging
from concurrent.futures import ProcessPoolExecutor

from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from hydra.utils import instantiate
from pathlib import Path
from nuplan.planning.script.builders.logging_builder import build_logger
from contextlib import asynccontextmanager

from .helper import score_single_trajectory, score_same_token_trajectory

hydra.initialize(config_path="config/pdm_scoring", version_base=None)
cfg = hydra.compose(config_name="default_run_pdm_score_server")
cfg.experiment_name = 'gunicorn_server'

build_logger(cfg)
logger = logging.getLogger(__name__)

# --- Preload resources ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading heavy resources...")
    app.state.cfg = cfg
    cfg.metric_cache_path=os.getenv('CACHE_PATH', cfg.metric_cache_path)
    app.state.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    logger.info(f"cfg.metric_cache_path: {cfg.metric_cache_path}")

    logger.info("Resources loaded successfully.")
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    token: str
    poses: List[List[float]]
    verbose: bool
    
class ScoreGroupRequest(BaseModel):
    token: str
    poses: List[List[List[float]]]
    verbose: Optional[bool] = False


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/score")
def score_endpoint(req: ScoreRequest):
    cfg = app.state.cfg
    metric_cache_loader = app.state.metric_cache_loader
    # metrics = score_single_trajectory(cfg, metric_cache_loader, simulator, scorer, req.token, req.poses)
    verbose = req.verbose if req.verbose is not None else False
    metrics = score_single_trajectory(cfg, metric_cache_loader, req.token, req.poses, verbose)
    for k in metrics:
        if isinstance(metrics[k], np.ndarray):
            metrics[k] = metrics[k].tolist()
    return metrics

@app.post("/score_group")
def score_same_token_endpoint(req: ScoreGroupRequest):
    cfg = app.state.cfg
    metric_cache_loader = app.state.metric_cache_loader

    # Same token, poses is a B x N x 3 array
    
    verbose = req.verbose if req.verbose is not None else False
    metrics = score_same_token_trajectory(cfg, metric_cache_loader, req.token, req.poses, verbose)
    # metrics: List[metrics_dict]
    for metrics_dict in metrics:
        for k in metrics_dict:
            if isinstance(metrics_dict[k], np.ndarray):
                metrics_dict[k] = metrics_dict[k].tolist()
    return metrics

    

# Keep main function for direct execution (e.g., debugging)
def main():
    # Launch a single uvicorn instance
    port = cfg.get('server', {}).get('port', 8900)
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()