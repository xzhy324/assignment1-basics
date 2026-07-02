import math

# glue code for `uv run pytest -k test_get_lr_cosine_schedule`
def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    lr_t = -1
    if it < warmup_iters:
        lr_t = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        cos_val = math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi)
        lr_t = min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + cos_val)
    else:
        lr_t = min_learning_rate
    return lr_t