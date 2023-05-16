import os
import numpy as np
from .config import Config
from pathlib import Path


def get_path(Config: Config, 
             filetype: str = "trajectory",
             overwrite: bool = True):
        """
        Creates an outfile str for a specified filetype that includes the absolute directory and filename. 
        If the directory does not exist, it is created. The overwrite call checks if the system name already 
        exists in the output directory. If it does, a version str ("_vXX") is appended to the system name. 
        
        Filetypes:
            "trajectory"
            "config"
            "init_pos"
            "parameters"
            "rdf"
            "structure_factor"
        """
        
        # if the system should not be overwritten change the version
        dir_dict = {"trajectory":       ("/trajectories/traj_",                 ".gro"),
                    "config":           ("/configs/cfg_",                       ".toml"),
                    "parameters":       ("/parameters/param_",                  ".toml"),
                    "init_pos":         ("/initial_positions/xyz_",             ".npy"),
                    "rdf":              ("/results/rdf/rdf_",                   ".npy"),
                    "structure_factor": ("/results/structure_factor/Sq_",       ".npy")}
        
        if overwrite == False:
            # check if any file already exists
            for rel_path, ftype in dir_dict.values():
                fname = Config.dir_sys + rel_path + Config.name_sys + ftype
                k = 0
                while os.path.exists(fname):
                    version = f"_v{k:d}" # update version until no files are found
                    fname = Config.dir_sys + rel_path + Config.name_sys + version + ftype
                    k += 1
                if k > 0:
                    version = f"_v{k-1:d}"
                else:
                    version = ""
        else:
            version = ""
        
        # create outfile string
        fname = Config.dir_sys + dir_dict[filetype][0] + Config.name_sys + version + dir_dict[filetype][1]
        
        # make dir if path doesnt exist
        path = Path(fname).parent
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        return fname