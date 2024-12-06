import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./../config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
DB_DIR = config["paths"]["dbs_dir"]
sys.path.append(PROJECT_DIR)

from src.objectives.warcraft import WarcraftObjective
# from src.samplers.custom import CustomQMCSampler, CustomRandomSampler
from src.samplers.parafac_v1 import ParafacSampler as ParafacSampler_v1
from src.samplers.parafac import ParafacSampler
from src.utils_experiments import search_log_files, set_logger

from src.samplers.custom_gp import CustomGPSampler

__all__ = [
    "WarcraftObjective",
    "ParafacSampler",
    "search_log_files",
    "set_logger",
    "CustomGPSampler",
]
