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
        
        self.nbeads             = len(self.positions)
        self.ntags              = len(set(self.tags))
        
        self._validate_input()
        
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
        assert self.epsilon_lj.shape        == (self.ntags, self.ntags), "Number of tags and number of bond types do not match"
        assert self.sigma_lj.shape          == (self.ntags, self.ntags), "Number of tags and number of bond types do not match"
                    
        return