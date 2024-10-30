import yaml
import logging
import torch
import numpy as np
from scipy.cluster.vq import kmeans
from .utils import (
    get_box_sizes_and_class_weights, 
    get_box_sizes_and_class_weights_from_polygons, 
)
from typing import *

logger = logging.getLogger(__name__)

def ratio_metrics(
        anchors: torch.Tensor,
        wh_data: torch.Tensor,  
        threshold: float=4.0
    ) -> torch.Tensor:
    # k shape: n x 2
    r = wh_data[:, None] / anchors[None]
    v = torch.min(r, 1/r).min(dim=2).values
    v = v.max(dim=1).values
    m = (v > 1 / threshold).float()
    scores = (v * m).mean()
    return scores.item()
    
def ratio_metrics_w_extras(
        anchors: torch.Tensor, 
        wh_data: torch.Tensor,
        threshold: float=4.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = wh_data[:, None] / anchors[None]
    v = torch.min(r, 1/r).min(dim=2).values
    v = v.max(dim=1).values
    m = (v > 1 / threshold).float()
    bpr = m.mean()                  # best possible recall
    aat = m.sum()                   # anchors above threshold
    score = (v * m).mean()
    return score.item(), bpr.item(), aat.item()

def cluster_anchors_w_mutation(
        wh_data: torch.Tensor, 
        num_anchors: int=9,
        threshold: int=4.0, 
        num_generations: int=100,
        kmeans_iter: float=30,
        verbose: bool=True,
        mut_proba: float=0.9,
        sigma: float=0.1,
    ) -> Tuple[torch.Tensor, float, float, float]:
    device = wh_data.device
    # wh_data shape: N x 2
    def log_generation(anchors: torch.Tensor, gen: Optional[int]=None, is_best_gen: bool=False):
        if verbose:
            anchors = anchors[torch.argsort(anchors.prod(1))]
            score, bpr, aat = ratio_metrics_w_extras(anchors, wh_data, threshold)
            score_str = "score" if not is_best_gen else "best score"
            print((f"Generation: {gen}, BRP: {bpr :.4f}, AAT: {aat :.4f} {score_str}={score :.4f}"))
        return
    try:
        print("Clustering anchors...")
        assert num_anchors <= len(wh_data)
        w_sigma = wh_data.std(0)  # sigmas for whitening
        solution, _ = kmeans(wh_data.cpu().numpy() / w_sigma.cpu().numpy(), num_anchors, iter=kmeans_iter)
        solution = torch.from_numpy(solution).to(device) * w_sigma
        # kmeans may return fewer points than requested if wh is insufficient or too similar
        assert num_anchors == solution.shape[0]
    except AssertionError:
        solution = torch.sort(torch.rand(num_anchors, 2))
    log_generation(solution)

    best_score = ratio_metrics(solution, wh_data, threshold)
    best_gen = None
    best_solution = solution
    
    tgrand = lambda *args : torch.rand(*args, device=device)
    tgrandn = lambda *args : torch.randn(*args, device=device)
    for gen in range(0, num_generations):
        mut_factor = torch.ones(*solution.shape)
        while (mut_factor == 1).all():
            mut_factor = ((tgrand(*solution.shape) > mut_proba) * tgrand(1).item() * tgrandn(solution.shape) * sigma) + 1
        new_solution = solution * mut_factor

        is_best_gen = False
        score = ratio_metrics(new_solution, wh_data, threshold)
        if score > best_score:
            best_gen = gen
            best_solution = solution
            best_score = score
            is_best_gen = True
        log_generation(new_solution, gen, is_best_gen)
    
    best_solution = best_solution[torch.argsort(best_solution.prod(dim=-1))]
    best_score, bpr, aat = ratio_metrics_w_extras(best_solution, wh_data, threshold)
    if verbose:
        print(f"best solution: {best_solution}")
        print(f"best score is {best_score :.4f} @ generation {best_gen}")
        print(f"Best Possible Recall: {bpr :.4f}")
        print(f"Anchors Above Threshold: {aat}")
    return best_solution, best_score, bpr, aat

def generate_anchors_and_class_weights(
        labels_path: str, 
        predefined_anchors: Dict[str, List[List[float]]],
        threshold: float=4.0, 
        score_tol: float=0.8, 
        bpr_tol: float=0.95,
        verbose: bool=True,
        update_anchors_cfg: bool=True,
        anchors_path: Optional[str]=None,
        from_polygons: bool=False,
        device: Union[str, int]="cpu",
        **kwargs
    ) -> torch.Tensor:
    predefined_anchors = [
        torch.tensor(predefined_anchors["sm"], device=device, dtype=torch.float32), 
        torch.tensor(predefined_anchors["md"], device=device, dtype=torch.float32), 
        torch.tensor(predefined_anchors["lg"], device=device, dtype=torch.float32)
    ]
    device = predefined_anchors[0].device
    num_anchors = len(predefined_anchors) * predefined_anchors[0].shape[0]
    anchors = torch.cat(predefined_anchors, dim=0)
    if not from_polygons:
        wh_data, class_weights = get_box_sizes_and_class_weights(labels_path)
    else:
        wh_data, class_weights = get_box_sizes_and_class_weights_from_polygons(labels_path)
    class_weights = torch.from_numpy(class_weights).to(device)
    wh_data = torch.from_numpy(wh_data).to(device)
    score, bpr, aat = ratio_metrics_w_extras(anchors, wh_data, threshold)
    if score >= score_tol and bpr >= bpr_tol:
        logger.info("Current anchors are a good fit for the dataset")
        anchors = anchors.reshape(3, 3, 2)
    else:
        logger.info("Current anchors are a poor fit for the dataset, attempting to improve: ")
        anchors, new_score, new_bpr, new_aat = cluster_anchors_w_mutation(
            wh_data, num_anchors, threshold, verbose=verbose, **kwargs
        )
        anchors = anchors.reshape(3, 3, 2)
        if new_score > score and new_bpr >= bpr:
            logger.info("Calculated anchors are a better fit than the previous anchors")
        if new_score > score_tol and new_bpr >= bpr_tol:
            logger.info("Calculated anchors are a good fit for the dataset")
        else:
            logger.info("Unfortunately, the calculated anchors are still a poor fit for the dataset")

        if update_anchors_cfg and anchors_path:
            with open(anchors_path, "r") as f:
                cfg = yaml.safe_load(f)
            f.close()
            with open(anchors_path, "w") as f:
                if "anchors" not in cfg:
                    cfg["anchors"] = {}
                cfg["anchors"]["sm"] = anchors[0].cpu().numpy().tolist()
                cfg["anchors"]["md"] = anchors[1].cpu().numpy().tolist()
                cfg["anchors"]["lg"] = anchors[2].cpu().numpy().tolist()
                yaml.safe_dump(cfg, f)
            f.close()
            logger.info(f"{anchors_path} has successfully been updated with the calculated anchors")
    anchors = anchors.to(dtype=torch.float32, device=anchors.device)
    return anchors, class_weights