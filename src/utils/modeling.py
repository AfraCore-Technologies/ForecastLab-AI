import pandas as pd


__all__ = ["model_selection"]




def model_selection(timeseries:pd.series, cutoffs: dict, models:dict) -> tuple[str,str]:
    """
    select the appropriate model based on demand caracteristics.

    input
    ----------
    timeseries: historical demand
    cutoffs:    thresholds for classification {"nzd": float, "cv2": float, "min_obs": int}
    models:     mapping from demand class to model name

    returns
    ----------
    tuple:      selected model name, class
    """
    min_obs = cutoffs.get("min_obs", 6)
    nzd_cutoff = cutoffs.get("nzd", 1/ 1.32)
    cv2_cutoff = cutoffs.get("cv2", 0.49)

    # New Item
    if timeseries is None or len(timeseries) < min_obs:
        return (models.get("new", "xgboost"), "new")
    
    non_zero_demand = timeseries.loc[timeseries != 0]

    # ALl zero
    if non_zero_demand.empty:
        return models.get("new", "xgboost"), "new"
    
    # non zero demand percentage(nzd)
    nzd = len(non_zero_demand) / len(timeseries)

    # squared coefficient of variation (cv2)
    mean_demand = timeseries.mean()
    if mean_demand == 0:
        return models.get("new", "xgboost"), "new"
    std_demand = timeseries.std(ddof=0)
    cv2 = (std_demand / mean_demand) ** 2 

    # classification
    if nzd < nzd_cutoff:
        if cv2 < cv2_cutoff:
            return models.get("intermittent", "xgboost"), "intermittent"
        else:
            return models.get("lumpy", "xgboost"), "lumpy"
    else:
        if cv2 < cv2_cutoff:
            return models.get("smooth", "xgboost"), "smooth"
        else:
            return models.get("erratic", "xgboost"), "erratic"
