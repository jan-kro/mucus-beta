import os
import toml
import numpy as np
from .config import Config
from typing import Optional
from pathlib import Path

class Topology:
    
    def __init__(self, Config: Config):
        
        self.positions = np.load(self._get_path(Config, "init_pos"))
        params = toml.load(open(self._get_path(Config, "parameters"), encoding="UTF-8"))
        
        self.r_bead             = np.array(params["rbeads"]).reshape(-1, 1)
        self.q_bead             = np.array(params["qbeads"]).reshape(-1, 1)
        self.mobility           = np.array(params["mobilities"]).reshape(-1, 1)
        self.force_constant_nn  = np.array(params["force_constants"])
        self.epsilon_lj         = np.array(params["epsilon_LJ"])
        self.sigma_lj           = np.array(params["sigma_LJ"])
        self.bonds              = np.array(params["bonds"])
        self.tags               = np.array(params["tags"])
        
        if "r0_bonds" not in params.keys():
            self.r0_bond        = 2*np.ones_like(self.force_constant_nn)
        else:
            self.r0_bond        = np.array(params["r0_bonds"])
        
        self.nbeads             = len(self.positions)
        self.ntags              = len(set(self.tags))
        
        self._validate_input()
    
    def set_parameter(self, value, type="epsilon_LJ"):
        
        if type == "epsilon_LJ":
            assert value.shape == self.epsilon_lj.shape, "Shape of new parameter does not match"
            self.epsilon_lj = value
        elif type == "sigma_LJ":
            assert value.shape == self.sigma_lj.shape, "Shape of new parameter does not match"
            self.sigma_lj = value
        elif type == "force_constants":
            assert value.shape == self.force_constant_nn.shape, "Shape of new parameter does not match"
            self.force_constant_nn = value
        elif type == "r0_bonds":
            assert value.shape == self.r0_bond.shape, "Shape of new parameter does not match"
            self.r0_bond = value
        elif type == "bonds":
            assert value.shape == self.bonds.shape, "Shape of new parameter does not match"
            self.bonds = value
        elif type == "tags":
            assert value.shape == self.tags.shape, "Shape of new parameter does not match"
            self.tags = value
        elif type == "rbeads":
            assert value.shape == self.r_bead.shape, "Shape of new parameter does not match"
            self.r_bead = value
        elif type == "qbeads":
            assert value.shape == self.q_bead.shape, "Shape of new parameter does not match"
            self.q_bead = value
        elif type == "mobilities":
            assert value.shape == self.mobility.shape, "Shape of new parameter does not match"
            self.mobility = value
        else:
            raise ValueError("Unknown parameter type")
        
    def save(self,
             config: Optional[Config]):
        """
        Saves the initial positions and the topology parameters to the output directory
        spcified in the config file.
        """
        
        positions_new = self.positions
        
        parameters_new = {}
        for attr in dir(self):
            # loop through all class variables (not functions etc)
            if not attr.startswith("__") and not callable(getattr(self,attr)):
                # only take vaariables that are contained in the parameters file
                if (attr != "positions") and (attr != "nbeads") and (attr != "ntags"):
                    value = getattr(self, attr).tolist()
                    parameters_new[attr] = value
            
        pos_exists = False
        param_exists = False
        filetpes = ["init_pos", "parameters"]
        for filetype in filetpes:
            filename = self._get_path(config, filetype, overwrite=True)
            
            # check if path already exists
            if os.path.exists(filename):
                if filetype == "init_pos":
                    pos_exists = True
                    # if path exists, check if the positions are the same
                    positions_old = np.load(filename)
                    if np.all(positions_old == positions_new):
                        # if positions are the same, do not save new positions
                        overwrite_pos = False
                    else:
                        # if positions are not the same, ask for permission to overwrite
                        print("Topology.save(): initial positions file already exists and new positions differ from old ones.")
                        overwrite_pos = "y" == input("Do you want to Overwrite? (y/n)")
                        if overwrite_pos:
                            np.save(filename, positions_new)
                else:
                    param_exists = True
                    #check if all parameters are the same
                    
                    # load old parameters
                    params_old = toml.load(open(filename, encoding="UTF-8"))

                    n_diff = 0 # counts number of parameters that are different
                    for key in parameters_new.keys():
                        if parameters_new[key] != params_old[key]:
                            n_diff +=1
                            
                    if n_diff != 0:
                        print("Topology.save(): parameter file already exists and new parameters differ from old ones.")
                        overwrite_param = "y" == input("Do you want to Overwrite? (y/n)")
                    else:
                        overwrite_param = False
                        if overwrite_param:
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
            # NOTE DOESNT WORK BECAUSE FILENAME IS ITERATED OVER                
            if not pos_exists or overwrite_pos:
                np.save(filename, positions_new)
            if not param_exists or overwrite_param:
                outstr_param = ""
                for key, item in parameters.items():
                    if item is not None:
                        outstr_param += key + "=" + str(item) + "\n"
                    
                f = open(filename, mode="w")
                f.write(outstr_param)
                f.close()
                    
        
            
    def get_tags(self, i, j):
        """
        Parameters:
            i, j (int):
                Bead indices, or lists of bead indices
        Returns:
            ti, tj (int):
                Tags/lists of tags correspondig to atom i and j
        """
        return self.tags[i], self.tags[j]
    
    # NOTE this function is completly unnecessary    
    def get_force_constant(self, i, j):
        """
		i, j are the indices of the atoms
		self.tags is an n_atoms long list of integers
		self.force_constant is an array of shape (n_tags, n_tangs) 
        containing the force constants for all interaction types
		"""
		
        idx1 = self.tags[i]
        idx2 = self.tags[j]
		
        return self.force_constant_nn[idx1, idx2]

    def _validate_input(self):
        
        assert len(self.r_bead)   == self.nbeads, "Dimensions of rbead and positions do not match"
        assert len(self.q_bead)   == self.nbeads, "Dimensions of qbead and positions do not match"
        assert len(self.mobility) == self.nbeads, "Dimensions of mobility and positions do not match"
        assert len(self.tags)     == self.nbeads, "Dimensions of tags and positions do not match"
        
        assert self.force_constant_nn.shape == (self.ntags, self.ntags), "Number of tags and number of bond types do not match"
        assert self.r0_bond.shape           == (self.ntags, self.ntags), "Number of tags and number of bond types do not match"
        assert self.epsilon_lj.shape        == (self.ntags, self.ntags), "Number of tags and number of bond types do not match"
        assert self.sigma_lj.shape          == (self.ntags, self.ntags), "Number of tags and number of bond types do not match"
        
        return
    
    def _get_path(self,
                  Config: Optional[Config], 
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
            "forces"
        """
        
        
        DIR_DICT = {"trajectory":           ("/trajectories/traj_",                 ".gro"),
                    "config":               ("/configs/cfg_",                       ".toml"),
                    "parameters":           ("/parameters/param_",                  ".toml"),
                    "init_pos":             ("/initial_positions/xyz_",             ".npy"),
                    "rdf":                  ("/results/rdf/rdf_",                   ".npy"),
                    "structure_factor":     ("/results/structure_factor/Sq_",       ".npy"),
                    "structure_factor_rdf": ("/results/structure_factor/Sq_rdf_",   ".npy"),
                    "stress_tensor":        ("/results/stress_tensor/sigma_",       ".npy"),
                    "msd":                  ("/results/msd/msd_",                   ".npy"),
                    "forces":               ("/forces/forces_",                     ".txt")}
        
        if isinstance(Config, str):
            fname = Config.split("/configs/cfg_")[0] + DIR_DICT[filetype][0] + Config.split("/configs/cfg_")[1].split(".")[0] + DIR_DICT[filetype][1]
        else:
            fname = Config.dir_sys + DIR_DICT[filetype][0] + Config.name_sys + DIR_DICT[filetype][1]
        
        # split fname into base, name, version and ext
        head_tail = os.path.split(fname)
        base = head_tail[0]
        name_tot, ext = os.path.splitext(head_tail[1])
        split_arg = "_v"

        # if the system should not be overwritten change the version
        name_version = name_tot.split(split_arg)
        
        # check if there is a version contained in the name
        if len(name_version) == 1:
            # system name does not contain split arg "_v"
            name = name_version[0]
            version = 0
        else:
            try:
                # make sure that version is an integer, otherwise it might just be a random "_vABC" string (idk eg. xyz_variable_input.npy)
                version = int(name_version[-1])
                name = name_version[0]
                # reassemble name in case there is a second "_v" in the name
                for part in name_version[1:-1]:
                    name += split_arg + part
            # if split_arg is in the name but not followed by an integer, use total name as name
            except:
                # use total name as name
                name = name_tot
                version = 0

        if overwrite == False:
            # check if file already exists
            if not os.path.exists(base + "/" + name + ext):
                fname = base + "/" + name + ext
            else:
                # if file exist update version until no file with that version exists
                while os.path.exists(base + "/" + name + split_arg + str(version) + ext):
                    version += 1
                # update fname with new version
                fname = base + "/" + name + split_arg + str(version) + ext

        # if parent path doesnt exist, create it
        if not os.path.exists(Path(fname).parent):
            os.makedirs(Path(fname).parent)
                
        return fname