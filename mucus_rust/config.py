import re
import numpy as np
import toml
from pydantic import BaseModel, root_validator
import os
import time
from pathlib import Path
from typing import Optional


class Config(BaseModel, arbitrary_types_allowed=True):
    """
    The config file contains all the information of the system and where everything is saved
    """
    
    steps:              int
    stride:             int
    n_beads:            int
    timestep:           float
    r0_nm:              float                   = 0.1905 # 0.68 # bead radius in nm
    cutoff_LJ:          float                   = 2.0
    lB_debye:           float                   = 36.737 # 3.077
    c_S:                float                   = 10.0
    cutoff_debye:       float                   = None
    lbox:               Optional[float]         = None
    pbc:                bool                    = True
    cutoff_pbc:         Optional[float]         = None
    write_traj:         bool                    = True
    write_forces:       bool                    = True
    write_distances:    bool                    = True
    cwd:                Optional[str]           = os.getcwd()
    name_sys:           Optional[str]           = None
    dir_sys:            Optional[str]           = None
    simulation_time:    Optional[float]         = None
    # bonds:              Optional[np.ndarray]    = None 

    # TODO add current time and date to properly track everything
    
    @classmethod
    def test(cls):
        path = Path(str(os.getcwd())).parent
        path = str(path) + "/tests/data/connected-S-mesh/configs/cfg_connected-S-mesh_v0.toml"
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)
    
    @classmethod
    def from_toml(cls, path):
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)
    
    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)
    
    @root_validator
    def default_values(cls, values):
        for key, item in values.items():
            
            if key == "cwd":
                values[key] = os.getcwd()
            
            # create output directory    
            if key == "dir_sys":
                # if outdir is not specified create folder in cdw
                if item is None:
                    
                    now = time.localtime()
                    now_str = f"{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m{now.tm_sec}s"
                    name = values["name_sys"]
                    
                    dir_out = os.getcwd() + f"/systems/sys_{name:s}"                        
                    os.makedirs(dir_out)
                    values[key] = dir_out
            
        return values
    
    @root_validator(pre=True)
    def validate_ndarrays(cls, values):
        """
        Iterates through the whole config dictionary
        
        for the bonds key, either a list, is accepted, which is then turend into a ndarray, or a str is accepted, which specifies a path leading to a saved numpy array
        """
        for key, item in values.items():
            data_type = cls.__annotations__[key]
            if data_type == Optional[np.ndarray] or data_type == np.ndarray:
                if item is not None:
                    if isinstance(item, str):
                        values[key] = np.load(item)
                    else:
                        values[key] = np.array(item)
                else:
                    if data_type is np.ndarray:
                        raise ValueError(f"We expected array for {key} but found None.")
            
        return values

    def __format__(self, __format_spec: str) -> str:
        """Format the config as a toml file such that atrributes
        are on multiple lines. Activate if __format_spec is 'fancy'
        """
        output = str(self)
        if __format_spec == "'fancy'" or __format_spec == "fancy":
            # find all spaces that are followed by an attribute of
            # the dataclass and replace them with a newline
            output = re.split("( [a-z|_]+=)", output)
            for i, item in enumerate(output):
                if item.startswith(" "):
                    output[i] = f"\n{item[1:]}"
            output = "".join(output)
            output = output.replace("=", " = ")
        return output
    
    def save_config(self, fout: str = None, overwrite: bool = True):
        """
        saves current self.config in a .toml file 
        """
        
        #TODO add overwrite option
        
        # check if different pathout is specified, than in config
        if fout is None:
            fout = self.dir_sys + "/configs/cfg_" + self.name_sys + ".toml"
        
        output = str(self)
        output = output.replace("=True", "=true") # for toml formating
        output = output.replace("=False", "=false")
        output = output.replace(" ", "\n")
        output = output.replace("=", " = ")

        f = open(fout, "w")
        f.write(output)
        f.close()
        
        return


if __name__ == "__main__":
    config = Config.from_toml("/home/jan/Documents/masterthesis/project/mucus/configs/tests/cfg_test_box_10_12_0.toml")
    print(config.dir_sys)
    print(config.fname_sys)
    print(config.fname_traj)


