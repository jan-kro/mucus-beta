import numpy as np
import toml
from .config import Config
from .utils import get_path

class Topology:
    
    def __init__(self, Config: Config):
        
        self.positions = np.load(get_path(Config, "init_pos"))
        params = toml.load(open(get_path(Config, "parameters"), encoding="UTF-8"))
        
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