import os
import numpy as np
from .system import System
from .config import Config
from .topology import Topology
from .utils import get_path, load_trajectory
import mdtraj as md


class Analysis:
    
    def __init__(self, 
                 cfg: Config, 
                 frame_range: tuple = None):
        
        self.cfg = cfg
        self.trajectory = load_trajectory(cfg, frame_range)
        self.topology = Topology(cfg)
        
    def rdf(self,
            r_range = None,
            n_bins = None,
            bin_width = 0.05,
            tags = (0,0),
            save=True,
            overwrite=False):
        
        if r_range is None:
            r_range = np.array([1.5, self.cfg.lbox/2])
            
        if n_bins is None:   
            n_bins = int((r_range[1] - r_range[0]) / bin_width)
        
        lbox = self.cfg.lbox
        natoms = len(self.trajectory[0])

        # create unic cell information
        uc_vectors = np.repeat([np.array((lbox, lbox, lbox))], len(self.trajectory), axis=0)
        #uc_angles = np.repeat([np.array((90,90,90))], len(self.trajectory), axis=0)
        
        
        # create mask for tags
        mask_1 = list(())
        mask_2 = list(())
        for i in range(natoms):
            mask_1.append(self.topology.tags[i] == tags[0])
            mask_2.append(self.topology.tags[i] == tags[1])
        
        # number of particles per mask
        n1 = np.sum(mask_1)
        n2 = np.sum(mask_2)
        
        # create bond pair list
        pairs = list()
        for i in range(natoms-1):
            for j in range(i+1, natoms):
                pairs.append((i, j))
        pairs = np.array(pairs)
        
        g_r = np.zeros(n_bins)
        
        for frame in self.trajectory:
            # make 3d verion of meshgrid
            r_left = np.tile(frame, (n2, 1, 1)) # repeats vector along third dimension len(a) times
            r_right = np.reshape(np.repeat(frame, n1, 0), (n2, n1, 3)) # does the same but "flipped"
            
            directions = r_left - r_right # this is right considering the mesh method. dir[i, j] = r_j - r_i

            # apply minimum image convetion
            directions -= self.cfg.lbox*np.round(directions/self.cfg.lbox)

            # calculate distances and apply interaction cutoff
            distances = np.linalg.norm(directions, axis=2)
            mask_dist = np.logical_and(r_range[0] < distances, distances < r_range[1]) 

            g_r_frame, edges = np.histogram(distances[mask_dist], range=r_range, bins=n_bins)
            g_r += g_r_frame
        r = 0.5 * (edges[1:] + edges[:-1])

        # Normalize by volume of the spherical shell.
        # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
        # a less biased way to accomplish this. The conclusion was that this could
        # be interesting to try, but is likely not hugely consequential. This method
        # of doing the calculations matches the implementation in other packages like
        # AmberTools' cpptraj and gromacs g_rdf.
        
        # unitcell_volumes = np.array(list(map(np.linalg.det, uc_vectors))) # this should be used if the unitcell is not cubic
        unitcell_volumes = np.prod(uc_vectors, axis=1)
        V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        norm = len(pairs) * np.sum(1.0 / unitcell_volumes) * V # the trajectory length is implicitly included in the uc volumes
        g_r = g_r.astype(np.float64) / norm
        
        number_density = len(pairs) * np.sum(1.0 / unitcell_volumes) / natoms / len(self.trajectory)
        
        if save == True:
            fname = get_path(self.cfg, filetype="rdf", overwrite=overwrite)
            np.save(fname, np.array([r, g_r]))
        
        return r, g_r, number_density
    
    def structure_factor(self,
                         g_r        = None,
                         radii      = None,    
                         rho        = None, 
                         qmax       = 2.0, 
                         n          = 1000,
                         save       = True,
                         overwrite  = False):

        """
        Compute structure factor S(q) from a radial distribution function g(r).
        The calculation of S(q) can be further simplified by moving some of 
        the terms outside the sum. I have left it in its current form so that 
        it matches S(q) expressions found commonly in books and in literature.

        Parameters
        ----------

        g_r : np.ndarray
            Radial distribution function, g(r).
        radii : np.ndarray
            Independent variable of g(r).
        rho : float
            .
        qmax : floatAverage number density of particles
            Maximum value of momentum transfer (the independent variable of S(q)).
        n : int
            Number of points in S(q).
        save : bool (default: True)
            Wether the [Q, S_q] should be saved in an .npy file
            
        Returns
        -------

        Q : np.ndarray
            Momentum transfer (the independent variable of S(q)).
        S_q : np.ndarray
            Structure factor
        """

        # calculate rdf, if not given
        if g_r is None:
            radii, g_r, rho = self.rdf(overwrite=overwrite)

        n_r = len(g_r)
        Q = np.linspace(0.0, qmax, n)
        S_q = np.zeros_like(Q)

        dr = radii[1] - radii[0]
            
        h_r = g_r - 1

        S_q[0] = np.sum(radii**2*h_r)

        for q_idx, q in enumerate(Q[1:]):
            S_q[q_idx+1] = np.sum(radii*h_r*np.sin(q*radii)/q)

        S_q = 1 + 4*np.pi*rho*dr * S_q / n_r

        if save == True:
            fname = get_path(self.cfg, filetype="structure_factor", overwrite=overwrite)
            np.save(fname, np.array([Q, S_q]))

        return Q, S_q