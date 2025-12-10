import torch


def topk_accuracy(logits, targets, k=5):
    """
    Calculate top-k accuracy (hitrate).
    
    Args:
        logits: Model predictions (batch_size, vocab_size)
        targets: Ground truth labels (batch_size,)
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as float
    """
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def precision_at_k(logits, targets, k=5):
    """
    Precision@K for single-label targets.

    With exactly one relevant item per example, Precision@K reduces to
    hitrate@K (is the target in the top-K?). This avoids under-reporting
    by dividing by K, which is appropriate only for multi-relevance cases.
    """
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def recall_at_k(logits, targets, k=5):
    """
    Calculate Recall@K metric.
    
    Recall@K = (# relevant items in top-K) / (# total relevant items)
    For single target, this equals hitrate (either 1 or 0).
    
    Args:
        logits: Model predictions (batch_size, vocab_size)
        targets: Ground truth labels (batch_size,)
        k: Number of top predictions to consider
    
    Returns:
        Average Recall@K across batch
    """
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def mean_reciprocal_rank(logits, targets):
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = Average of (1 / rank of first relevant item)
    Higher rank (closer to 1) means better performance.
    
    Args:
        logits: Model predictions (batch_size, vocab_size)
        targets: Ground truth labels (batch_size,)
    
    Returns:
        MRR score as float
    """
    # Get indices that would sort logits in descending order
    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    
    # Find rank of target (where target appears in sorted list)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    
    # Calculate reciprocal ranks
    reciprocal_ranks = 1.0 / ranks.float()
    
    return reciprocal_ranks.mean().item()


def ndcg_at_k(logits, targets, k=5):
    """
    Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).
    
    NDCG accounts for position: items ranked higher contribute more.
    For binary relevance (0 or 1), DCG = sum(rel_i / log2(i+1))
    
    Args:
        logits: Model predictions (batch_size, vocab_size)
        targets: Ground truth labels (batch_size,)
        k: Number of top predictions to consider
    
    Returns:
        Average NDCG@K across batch
    """
    batch_size = logits.size(0)
    _, topk_indices = logits.topk(k, dim=1)
    
    # Create relevance matrix (1 if predicted item is target, 0 otherwise)
    relevance = topk_indices.eq(targets.unsqueeze(1)).float()
    
    # Calculate DCG: sum of rel_i / log2(i+2) for i in [0, k)
    # i+2 because: i starts at 0, and we want log2(2) for first position
    positions = torch.arange(1, k+1, device=logits.device).float()
    discounts = torch.log2(positions + 1)
    dcg = (relevance / discounts).sum(dim=1)
    
    # Ideal DCG (IDCG): best possible ranking (target at position 1)
    # For single relevant item, IDCG = 1 / log2(2) = 1
    idcg = 1.0
    
    # NDCG = DCG / IDCG
    ndcg = dcg / idcg
    
    return ndcg.mean().item()


def evaluate_all_metrics(logits, targets, k=5):
    """
    Calculate all evaluation metrics at once.
    
    Args:
        logits: Model predictions (batch_size, vocab_size)
        targets: Ground truth labels (batch_size,)
        k: Number of top predictions to consider
    
    Returns:
        Dictionary with all metric scores
    """
    metrics = {
        f'top{k}_accuracy': topk_accuracy(logits, targets, k),
        f'precision@{k}': precision_at_k(logits, targets, k),
        f'recall@{k}': recall_at_k(logits, targets, k),
        'mrr': mean_reciprocal_rank(logits, targets),
        f'ndcg@{k}': ndcg_at_k(logits, targets, k)
    }
    return metrics
