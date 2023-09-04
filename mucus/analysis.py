import os
import numpy as np
from .system import System
from .config import Config
from .topology import Topology
from .utils import get_path, load_trajectory, load_forces


class Analysis:
    
    def __init__(self, 
                 cfg: Config, 
                 frame_range: tuple = None,
                 stride: int = 1):
        
        self.cfg = cfg
        self.frame_range = frame_range
        self.stride = stride
        self.trajectory = load_trajectory(cfg, frame_range=frame_range, stride=stride)
        self.topology = Topology(cfg)
        self.system = System(cfg)
        
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

        # ALLEN TILDERSLEY METHOD:
        
        # g_r[i] is average number of atoms which lie to a distance between r[i] and r[i]+dr to each other
        g_r = g_r/natoms/len(self.trajectory)
        
        # number density of particles
        number_density = natoms/self.cfg.lbox**3 # NOTE only for cubical box
        
        # now normalize by shell volume
        shell_volumes = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        
        # normalisation
        g_r = g_r / shell_volumes / number_density
        
        # MDTRAJ METHOD:
        # Normalize by volume of the spherical shell.
        # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
        # a less biased way to accomplish this. The conclusion was that this could
        # be interesting to try, but is likely not hugely consequential. This method
        # of doing the calculations matches the implementation in other packages like
        # AmberTools' cpptraj and gromacs g_rdf.
        
        # # unitcell_volumes = np.array(list(map(np.linalg.det, uc_vectors))) # this should be used if the unitcell is not cubic
        # unitcell_volumes = np.prod(uc_vectors, axis=1)
        # V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        # norm = len(pairs) * np.sum(1.0 / unitcell_volumes) * V # the trajectory length is implicitly included in the uc volumes
        # g_r = g_r.astype(np.float64) / norm
        
        # number_density = len(pairs) * np.sum(1.0 / unitcell_volumes) / natoms / len(self.trajectory)
        
        if save == True:
            fname = get_path(self.cfg, filetype="rdf", overwrite=overwrite)
            np.save(fname, np.array([r, g_r]))
            with open(fname[:-4] + ".txt", "w") as f:
                f.write(f"rdf array with shape [2, {len(g_r)}] corresponding to [[r], [g(r)]]\n")
                if self.frame_range is not None:
                    f.write(f"frame range: {self.frame_range[0]:d} - {self.frame_range[1]:d}\n")
                else:
                    f.write(f"frame range: all\n")
                f.write(f"stride: {self.stride:d}\n")
                f.write(f"r_range: {r_range[0]:f} - {r_range[1]:f}\n")
                f.write(f"n_bins: {n_bins:d}\n")
                f.write(f"bin_width: {bin_width:f}\n")
                f.write(f"tags: ({tags[0]:d}, {tags[1]:d})\n")
        
        return r, g_r, number_density
    
    def structure_factor_rdf(self,
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
            fname = get_path(self.cfg, filetype="structure_factor_rdf", overwrite=overwrite)
            np.save(fname, np.array([Q, S_q]))

        return Q, S_q
    
    
    def structure_factor(self,
                         qmax = 2, 
                         n = 100,
                         traj = None,
                         save = True):
        """directly from trajectory"""
        
        if traj is None:
            traj = self.trajectory
            
        n_frames = len(traj)
        n_atoms = len(traj[0])
        lbox = self.cfg.lbox
        report_stride = int(n_frames//10)
        
        Q = np.linspace(0.0, qmax, n)
        S_q = np.zeros_like(Q)
        S_q[0] = n_atoms**2*n_frames/2
        
        print(f"Calculating structure factor for system {self.cfg.name_sys} ...\n")
        print("Frame    of    Total")
        print("--------------------")
        
        for k, frame in enumerate(traj):
            # make 3d verion of meshgrid
            r_left = np.tile(frame, (n_atoms, 1, 1)) # repeats vector along third dimension len(a) times
            r_right = np.reshape(np.repeat(frame, n_atoms, 0), (n_atoms, n_atoms, 3)) # does the same but "flipped"
            
            directions = r_left - r_right # this is right considering the mesh method. dir[i, j] = r_j - r_i

            # apply minimum image convetion
            directions -= lbox*np.round(directions/lbox)

            # calculate distances and only takes upper triangular matrix
            distances = np.linalg.norm(directions, axis=2)[np.triu_indices(n_atoms, k = 1)]
            
            for n, q in enumerate(Q[1:]):
                S_q[n+1] += np.sum(np.sin(q*distances)/(q*distances))
            
            if k%report_stride == 0:
                print(f"{k:<8d} of {n_frames:8d}")                
        print("done\n")
        
        if save:
            fname = get_path(self.cfg, filetype="structure_factor", overwrite=False)
            np.save(fname, np.array([Q, S_q]))
            with open(fname[:-4] + ".txt", "w") as f:
                f.write(f"structure factor array with shape [2, {len(S_q)}] corresponding to [[q], [S(q)]]\n")
                if self.frame_range is not None:
                    f.write(f"frame range: {self.frame_range[0]:d} - {self.frame_range[1]:d}\n")
                else:
                    f.write(f"frame range: all\n")
                f.write(f"stride: {self.stride:d}\n")
                f.write(f"qmax: {qmax:f}\n")
                f.write(f"number of S(q) values: {n:d}\n")
        return Q, 2*S_q/n_frames/n_atoms 
    
    def virial_stress_tensor(self,
                             save = True,
                             return_all = False,
                             particle_type = 0):
        # sigma = np.zeros((3,3)) # stress tensor reduced units with dimensionality of M/L/T^2
        sigma_t = np.zeros((len(self.trajectory), 3,3)) 
        volume = self.cfg.lbox**3                                                                        # reduced units (volume of box)

        # find indices of particle type
        if particle_type is not None:
            mask_frame = self.topology.tags == particle_type
            mask = np.tile(mask_frame, (len(self.trajectory), 3, 1)).transpose(0, 2, 1)
        else:
            mask = np.ones(len(self.trajectory), len(self.topology.tags), 3, dtype=bool)
        
        # center box around zero
        traj = self.trajectory[mask] - np.array([self.cfg.lbox/2, self.cfg.lbox/2, self.cfg.lbox/2])

        # TODO maybe do this in the setup
        forces_traj = load_forces(self.cfg, frame_range=self.frame_range)[mask]
        
        # calculate upper triangle of stress tensor
        for i, forces in enumerate(forces_traj):
            for a in range(3):
                for b in range(a, 3):
                    s = np.sum(traj[i,:,a] * forces[:,b])
                    # sigma[a,b] += s
                    sigma_t[i,a,b] += s

        
        kB = 1.380649e-23 # m^2 kg s^-2 K^-1
        T = 300 # K
        natoms = len(traj[0])
        red2m = self.system.r0_beeds_nm*1e-9 # m
        
        # divide by volume and convert to SI units
        # sigma = sigma/(volume*len(traj)*red2m*2) + np.eye(3)*kB*T*natoms/(volume*red2m**3)          # Pa -> SI units (stress tensor)
        sigma_t = sigma_t/(volume*red2m*2) + np.eye(3)*kB*T*natoms/(volume*red2m**3)


        # copy off-diagonal elements
        # for i in range(3):
        #     for j in range(i+1, 3):
        #         sigma[j,i] = sigma[i,j]
                
        for k in range(len(traj)):
            for i in range(3):
                for j in range(i+1, 3):
                    sigma_t[k,j,i] = sigma_t[k,i,j]
        
        # # get timestep
        # # TODO make this a utils function
        # mu = self.system.topology.mobility[0][0] # mobility of the system in reduced units
        # a = 1e-9*self.system.r0_beeds_nm/2       # m, reduced legth scale: PEG monomere radius
        # r = 1*a                        # m, bead radius

        # eta_w = 8.53e-4 # Pa*s
        # kB = 1.380649e-23 # m^2 kg s^-2 K^-1
        # T = 300 # K
        # mu_0 = kB*T/(6*np.pi*eta_w*r*a**2) 

        # dt_step = mu/mu_0
        # dt = self.cfg.stride*dt_step # s
        
        if save:
            fname = get_path(self.cfg, filetype="stress_tensor", overwrite=False)
            np.save(fname, sigma_t)
            with open(fname[:-4] + ".txt", "w") as f:
                f.write(f"stress tensor array with shape {sigma_t.shape} corresponding to [step, i, j]\n")
                if self.frame_range is not None:
                    f.write(f"frame range: {self.frame_range[0]:d} - {self.frame_range[1]:d}\n")
                else:
                    f.write(f"frame range: all\n")
                f.write(f"stride: {self.stride:d}\n")
                f.write(f"used particle type: {particle_type:d}\n")
        
        if return_all:
            return sigma_t
    
    def calculate_all(self,
                      save = True,
                      return_all = False):
        """
        calculates all properties and saves it into numpy arrays
        """
        print(f"Calculating all properties for system {self.cfg.name_sys} ...\n")
        print("Calculating rdf ...")
        r, g_r, number_density = self.rdf(save=save)
        print("Calculating structure factor ...")
        Q, S_q = self.structure_factor(save=save)
        print("Calculating stress tensor ...")
        sigma_t = self.virial_stress_tensor(save=save)
        print("done\n")
        
        if return_all:
            return r, g_r, number_density, Q, S_q, sigma_t
        
    def _correlation(self, a,b=None,subtract_mean=False):
        """
        Henriks acf code
        """

        meana = int(subtract_mean)*np.mean(a)

        a2 = np.append(a-meana, np.zeros(2**int(np.ceil((np.log(len(a))/np.log(2))))-len(a)))

        data_a = np.append(a2, np.zeros(len(a2)))

        fra = np.fft.fft(data_a)

        if b is None:

            sf = np.conj(fra)*fra

        else:

            meanb = int(subtract_mean)*np.mean(b)

            b2 = np.append(b-meanb, np.zeros(2**int(np.ceil((np.log(len(b))/np.log(2))))-len(b)))

            data_b = np.append(b2, np.zeros(len(b2)))

            frb = np.fft.fft(data_b)

            sf = np.conj(fra)*frb

        res = np.fft.ifft(sf)

        cor = np.real(res[:len(a)])/np.array(range(len(a),0,-1))

        return cor
        