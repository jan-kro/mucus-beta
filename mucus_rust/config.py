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
    n_particles:        int
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
    chunksize:          int                     = 1000 # number of frames per chunk in hdf5 file (1000 gives a 24MB chunk for 1000 particles)
    write_forces:       bool                    = True
    write_distances:    bool                    = True
    cwd:                Optional[str]           = os.getcwd()
    name_sys:           Optional[str]           = None
    dir_sys:            Optional[str]           = None
    simulation_time:    Optional[float]         = None

    
    # TODO add current time and date to properly track everything (maybe, because it is aleady in the file properties)
    
    @classmethod
    def test(cls):
        path = Path(str(os.getcwd())).parent
        path = str(path) + "/tests/data/connected-S-mesh/configs/cfg_connected-S-mesh_v0.toml"
        data = toml.load(open(path, encoding="UTF-8"))
        return cls(**data)
    
    @classmethod
    def from_toml(cls, path):
        
        if path == "test":
            path = "/net/storage/janmak98/masterthesis/output/test_systems/configs/cfg_mesh_tracer_6a_uncharged.toml"
        
        data = toml.load(open(path, encoding="UTF-8"))

        name_sys = data["name_sys"]
        name_sys_cfg = path.split("/")[-1].split(".")[0].split("cfg_")[0]
        
        if name_sys != name_sys_cfg and len(name_sys_cfg) > 0:
            proceed = "y" == input(f"Name of system in config file ({name_sys_cfg}) does not match name of system in config ({name_sys}). Proceed? (y/n)\n")
            if not proceed:
                change = "y" == input(f"Change name in config file from {name_sys} to {name_sys_cfg}? (y/n)\n")
                if change:
                    data["name_sys"] = name_sys_cfg
                else:
                    raise KeyboardInterrupt
            # raise ValueError(f"Name of system in config file ({name_sys_cfg}) does not match name of system in config ({name_sys})")
        
        return cls(**data)
    
    @classmethod
    def from_dict(cls, dict):
        # todo validate dict
        return cls(**dict)
    
    @root_validator(pre=True)
    def _default_values(cls, values):
        
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
        # backwards compatibility   
        if "n_beads" in values:
            values["n_particles"] = values["n_beads"]
            values.pop("n_beads")
            
        return values
    
    @root_validator(pre=True)
    def _validate_ndarrays(cls, values):
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
    
    @root_validator(pre=True)
    def _clean_config_file(cls, values):#, fout: str = None, overwrite: bool = True):
        """
        saves all current values, which have been passed through 
        the root validator funcitons in the .toml file, where they 
        were originally taken from. 
        """
        # config path
        fout = values["dir_sys"] + "/configs/cfg_" + values["name_sys"] + ".toml"
        # format output
        output = ""
        for key, item in values.items():
            if isinstance(item, str):
                output += f"{key} = '{item}'\n"
            else:
                if isinstance(item, np.ndarray):
                    output += f"{key} = '{item.tolist()}'\n"
                else:
                    output += f"{key} = {item}\n"
        output = output.replace("= True", "= true") # for toml formating
        output = output.replace("= False", "= false")
        
        # write
        f = open(fout, "w")
        f.write(output)
        f.close()
        
        return values
    
    def save(self, fout: str = None, overwrite: bool = True):
        """
        saves all current values, in a .toml file with the specified filepath fout.
        """
        # config path
        if fout is None:
            fout = self.dir_sys + "/configs/cfg_" + self.name_sys + ".toml"
        
        # TODO implement overwrite
        
        # format output
        output = ""
        for key, item in self.dict().items():
            if isinstance(item, str):
                output += f"{key} = '{item}'\n"
            else:
                output += f"{key} = {item}\n"
        output = output.replace("= True", "= true")
        output = output.replace("= False", "= false")

        # write
        f = open(fout, "w")
        f.write(output)
        f.close()

if __name__ == "__main__":
    # config = Config.from_toml("/net/storage/janmak98/masterthesis/output/test_systems/configs/cfg_mesh_tracer_6a_uncharged.toml")
    config = Config.from_toml("/net/storage/janmak98/masterthesis/output/mesh_talk/configs/cfg_mesh_tracer_1a_2.toml")
    print(config.dir_sys)
    print(config.name_sys)
    print(config.n_particles)


