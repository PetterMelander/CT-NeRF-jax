from pathlib import Path



def get_data_dir() -> Path:
    return Path(__file__).parents[1] / "data"

def get_model_dir() -> Path:
    return Path(__file__).parents[1] / "models"
