import os
import toml
import numpy as np
from .config import Config
from typing import Optional
from pathlib import Path

class Topology:
    
    def __init__(self, Config: Config):
        
        self.positions = np.load(self._get_path(Config, "init_pos"))
        params = self._load_params(Config)
        
        self.r_particle         = np.array(params["r_particles"])
        self.q_particle         = np.array(params["q_particles"])
        self.mobility           = np.array(params["mobilities"])
        self.force_constant_nn  = np.array(params["force_constants"])
        self.epsilon_lj         = np.array(params["epsilon_LJ"])
        self.sigma_lj           = np.array(params["sigma_LJ"])
        #self.bonds              = np.array(params["bonds"])      # TODO remove at some point
        self.bond_table         = np.array(params["bond_table"], dtype=bool)
        self.tags               = np.array(params["tags"],       dtype=np.uint)
        
        if "r0_bonds" not in params.keys():
            self.r0_bond        = 2*np.ones_like(self.force_constant_nn)
        else:
            self.r0_bond        = np.array(params["r0_bonds"])
        
        self.n_particles        = len(self.positions)
        self.ntags              = len(set(self.tags))
        
        self._validate_input()
        self._clean_parameter_file(Config)
        # self.save(Config)
    
    def _load_params(self, Config: Config):
        params = toml.load(open(self._get_path(Config, "parameters"), encoding="UTF-8"))
        
        # BACKWARDS COMPATIBILITY
        if "rbeads" in params.keys():
            params["r_particles"] = params["rbeads"]
            params.pop("rbeads")
            
        if "qbeads" in params.keys():
            params["q_particles"] = params["qbeads"]
            params.pop("qbeads")
        
        if 'bond_table' not in params.keys():
            n_particles = len(self.positions)
            bond_table = np.zeros((n_particles, n_particles), dtype=bool)
            for ij in params['bonds']:
                bond_table[ij[0], ij[1]] = True
            params['bond_table'] = bond_table
            params.pop('bonds')
            
        tags = params["tags"]
        n_tags = len(set(tags))
        
        if len(params["r_particles"]) != n_tags:
            r_all = np.array(params["r_particles"])
            r_particle = np.zeros(n_tags)
            for i, r in zip(tags, r_all):
                r_particle[i] = r
            params["r_particles"] = r_particle.tolist()
        
        if len(params["q_particles"]) != n_tags:
            q_all = np.array(params["q_particles"])
            q_particle = np.zeros(n_tags)
            for i, q in zip(tags, q_all):
                q_particle[i] = q
            params["q_particles"] = q_particle.tolist()
        
        if len(params["mobilities"]) != n_tags:
            mob_all = np.array(params["mobilities"])
            mob_particle = np.zeros(n_tags)
            for i, mob in zip(tags, mob_all):
                mob_particle[i] = mob
            params["mobilities"] = mob_particle.tolist()
        
        return params
    
    def _clean_parameter_file(self, config: Config):#, fout: str = None, overwrite: bool = True):
        """
        saves all current values, which have been passed through 
        the root validator funcitons in the .toml file, where they 
        were originally taken from. 
        """
        
        # params path
        fout = config.dir_sys + "/parameters/param_" + config.name_sys + ".toml"
        # format output
        
        KEY_DICT = self._get_key_dict()
        
        not_params = ["positions", "n_particles", "ntags"]
        parameters_new = {}
        for attr in dir(self):
            # loop through all class variables (not functions etc)
            if not attr.startswith("__") and not callable(getattr(self,attr)):
                # only take vaariables that are contained in the parameters file
                if attr not in not_params:
                    value = getattr(self, attr).tolist()
                    parameters_new[KEY_DICT[attr]] = value
        
        output = ""
        for key, item in parameters_new.items():
            if isinstance(item, str):
                output += f"{key} = '{item}'\n"
            else:
                if isinstance(item, np.ndarray):
                    output += f"{key} = '{item.tolist()}'\n"
                else:
                    output += f"{key} = {item}\n"
        output = output.replace("True", "true") # for toml formating
        output = output.replace("False", "false")
        
        # write
        f = open(fout, "w")
        f.write(output)
        f.close()
    
    def set_parameter(self, value, key="epsilon_LJ"):
        
        if key == "epsilon_LJ":
            assert value.shape == self.epsilon_lj.shape, f"Shape of new parameter '{key}' does not match"
            self.epsilon_lj = value
        elif key == "sigma_LJ":
            assert value.shape == self.sigma_lj.shape, f"Shape of new parameter '{key}' does not match"
            self.sigma_lj = value
        elif key == "force_constants":
            assert value.shape == self.force_constant_nn.shape, f"Shape of new parameter '{key}' does not match"
            self.force_constant_nn = value
        elif key == "r0_bonds":
            assert value.shape == self.r0_bond.shape, f"Shape of new parameter '{key}' does not match"
            self.r0_bond = value
        elif key == "bond_table":
            assert value.shape == self.bond_table.shape, f"Shape of new parameter '{key}' does not match"
            self.bond_table = value
        elif key == "tags":
            assert value.shape == self.tags.shape, f"Shape of new parameter '{key}' does not match"
            self.tags = value
        elif key == "r_particles":
            assert value.shape == self.r_particle.shape, f"Shape of new parameter '{key}' does not match"
            self.r_particle = value
        elif key == "q_particles":
            assert value.shape == self.q_particle.shape, f"Shape of new parameter '{key}' does not match"
            self.q_particle = value
        elif key == "mobilities":
            assert value.shape == self.mobility.shape, f"Shape of new parameter '{key}' does not match"
            self.mobility = value
        else:
            raise ValueError(f"Unknown parameter type: {key}")
    
    def _get_key_dict(self):
        KEY_DICT = {"r_particle": "r_particles",
                    "q_particle": "q_particles",
                    "mobility": "mobilities",
                    "force_constant_nn": "force_constants",
                    "epsilon_lj": "epsilon_LJ",
                    "sigma_lj": "sigma_LJ",
                    "bond_table": "bond_table",
                    "tags": "tags",
                    "r0_bond": "r0_bonds"}
        return KEY_DICT
       
    def save(self,
             config: Optional[Config]):
        """
        Saves the initial positions and the topology parameters to the output directory
        spcified in the config file.
        """
        
        KEY_DICT = self._get_key_dict()
        
        positions_new = self.positions
        
        parameters_new = {}
        for attr in dir(self):
            # loop through all class variables (not functions etc)
            if not attr.startswith("__") and not callable(getattr(self,attr)):
                # only take vaariables that are contained in the parameters file
                if (attr != "positions") and (attr != "n_particles") and (attr != "ntags"):
                    value = getattr(self, attr).tolist()
                    parameters_new[KEY_DICT[attr]] = value
            
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
                if filetype == "parameters":
                    param_exists = True
                    #check if all parameters are the same
                    
                    # load old parameters
                    params_old = toml.load(open(filename, encoding="UTF-8"))

                    n_diff = 0 # counts number of parameters that are different
                    for key in parameters_new.keys():
                        if parameters_new[key] != params_old[key]:
                            print("parameters of key " + key + " differ:")
                            print(parameters_new[key]) 
                            print(params_old[key])
                            n_diff +=1
                            
                    if n_diff != 0:
                        print("Topology.save(): parameter file already exists and new parameters differ from old ones.")
                        overwrite_param = "y" == input("Do you want to Overwrite? (y/n)")
                    else:
                        # if parameters are the same, do not save new parameters
                        overwrite_param = False
                                
            if ((not pos_exists) or overwrite_pos) and (filetype == "init_pos") :
                np.save(filename, positions_new)
            if ((not param_exists) or overwrite_param) and (filetype == "parameters"):
                outstr_param = ""
                for key, item in parameters_new.items():
                    if item is not None:
                        outstr_param += key + "=" + str(item) + "\n"
                    
                f = open(filename, mode="w")
                f.write(outstr_param)
                f.close()
                    
        
            
    def get_tags(self, i, j):
        """
        Completeley unnecessary function, but I am too lazy to change it everywhere
        
        Parameters:
            i, j (int):
                _particle indices, or lists of _particle indices
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
        
        assert len(self.r_particle) == self.ntags or len(self.r_particle) == self.n_particles, "Dimensions of r_particle and positions do not match"
        assert len(self.q_particle) == self.ntags or len(self.q_particle) == self.n_particles, "Dimensions of q_particle and positions do not match"
        assert len(self.mobility)   == self.ntags or len(self.mobility)   == self.n_particles, "Dimensions of mobility and positions do not match"
        
        assert len(self.tags)       == self.n_particles , "Dimensions of tags and positions do not match"
        
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