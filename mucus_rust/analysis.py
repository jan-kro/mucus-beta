import os
import h5py
import toml
import numpy as np
from .system import System
from .config import Config
from .topology import Topology
from .utils import get_path, load_trajectory_h5, load_results, load_forces, _validate_frame_range, get_number_of_frames, ResultsFiletypes
# TODO import .utils as utils
import datetime
import tidynamics as tid # TODO replace with own implementation


class Analysis:
    
    def __init__(self, 
                 cfg: Config, 
                 frame_range: list = None,
                 stride: int = 1):
        
        self.cfg = cfg
        self.frame_range = _validate_frame_range(frame_range, get_number_of_frames(cfg))
        self.stride = stride
        self.n_frames = len(np.arange(self.frame_range[0],self.frame_range[1],self.stride))
        # TODO 
        #! do not load whole traj into memory wtf
        self.trajectory = load_trajectory_h5(cfg, frame_range=self.frame_range, stride=stride)
        self.topology = Topology(cfg)
        self.system = System(cfg)
        self.time = np.arange(self.frame_range[0],self.frame_range[1],self.stride)*self.cfg.stride*self.cfg.timestep # reduced units
    
    def _get_key_dict(self):
        """
        TODO UNUSED PROBABLY DELETE
        """
        key_dict = {"rdf": [],
                    "structure_factor": [],
                    "structure_factor_rdf": [],
                    "stress_tensor": [],
                    "msd": [],
                    "forces": []}
        return key_dict
    
    def _exists(self,
                key_dict = {"key": "structure_factor",
                            "params": {"qmax": 2.0, "n": 1000}}):
        """
        Checks if a certain calculation was already done
        """
        # add frame range and stride to parameters
        key_dict["params"]["frame_range"] = self.frame_range
        key_dict["params"]["stride"] = self.stride
        
        fname = get_path(self.cfg, filetype="results", overwrite=True)
        with h5py.File(fname, "a") as fh5:
            key_params = key_dict["key"] + "_params"
            if key_params in fh5.keys():
                if fh5[key_params][()].decode("utf-8") == toml.dumps(key_dict["params"]):
                    return True
                else:
                    return False
        
    def _save(self,
              key_dict = {"key": "foo",
                          "params": {"a": None, "b": None}},
              data = None,
              overwrite = True):

        # add frame range and stride to parameters
        key_dict["params"]["frame_range"] = self.frame_range
        key_dict["params"]["stride"] = self.stride
        
        fname = get_path(self.cfg, filetype="results", overwrite=overwrite)
        with h5py.File(fname, "a") as fh5:
            while True:
                # check if parameters of spectified calculation already exist
                if key_dict["key"] + "_params" in fh5.keys():
                    # if they do exist, check if parameters are the same
                    if fh5[key_dict["key"] + "_params"][()].decode("utf-8") != toml.dumps(key_dict["params"]):
                        # ich the same dataset uses different parameters, ask if they should be overwritten
                        print(f"{key_dict['key']} already exists in the dataset, using different parameters.")
                        print(f"Old parameters:\n{fh5[key_dict['key'] + '_params'][()].decode('utf-8')}")
                        print(f"New parameters:\n{toml.dumps(key_dict['params'])}")
                        overwrite = "y" == input("Do you want to overwrite it? [y/n]: ")
                        if overwrite:
                            del fh5[key_dict["key"] + "_params"]
                            del fh5[key_dict["key"]]
                            break
                        # if they should not be overwritten ask for new name
                        print(f"The following keys are already in the dataset: {[key for key in fh5.keys() if '_params' not in key]}")
                        key_dict["key"] = input("Please enter a new key: ")
                        if key_dict["key"] not in fh5.keys():
                            break
                    # id dataset alreadty exists and parameters used for calculation are the same, ask if it should be overwritten or if the calculation should be skipped
                    else:
                        print(f"File {fname} already contains {key_dict['key']} with the same parameters.")
                        overwrite = "y" == input("Do you want to overwrite the dataset anyways? [y/n]: ")
                        if overwrite:
                            del fh5[key_dict["key"] + "_params"]
                            del fh5[key_dict["key"]]
                            break
                        else:
                            # if old calculation should not be overwritten, ask if it should be saved under a different name
                            save = "y" == input("Do you want to save the new calculation under a different name? [y/n]: ")
                            if save:
                                print(f"The following keys are already in the dataset: {[key for key in fh5.keys() if '_params' not in key]}")
                                key_dict["key"] = input("Please enter a new key: ")
                                if key_dict["key"] not in fh5.keys():
                                    break
                            else:
                                return
                else:
                    break
            
            fh5.create_dataset(key_dict["key"] + "_params", data=toml.dumps(key_dict["params"]))
            fh5.create_dataset(key_dict["key"], data=data)
        
    def rdf(self,
            r_range = None,
            n_bins = None,
            bin_width = 0.05,
            tags = (0,0),
            save=True,
            overwrite=False,
            return_all=False):
        
        key_dict = {"key": "rdf",
                    "params": {"r_range": r_range, 
                               "n_bins": n_bins,
                               "bin_width": bin_width,
                               "tags": tags}}
        
        if self._exists(key_dict):
            fname = get_path(self.cfg, filetype="results", overwrite=overwrite)
            with h5py.File(fname, "a") as fh5:
                key = key_dict["key"]
                r = fh5[key][:,0]
                g_r = fh5[key][:,1]
                return r, g_r
        
        # TODO RDF IS BROKEN FOR MULTIPLE PARTICLE TYPES
        
        if not save and not return_all:
            raise ValueError("Either save or return_all has to be True")
        
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
        # create mask for distances
        mask_1 = self.topology.tags == tags[0]
        mask_2 = self.topology.tags == tags[1]
        mask_1, mask_2 = np.meshgrid(mask_1,mask_2)
        mask_pairs = np.logical_and(np.logical_and(mask_1, mask_2), np.logical_not(np.eye(self.cfg.n_particles, dtype=bool)))
        
        g_r = np.zeros(n_bins)
        
        print(f"Calculating rdf for system {self.cfg.name_sys} ...\n")
        print("Started at ", datetime.datetime.now(), "\n")
        print("Frame    of    Total")
        print("--------------------")
        report_stride = int(len(self.trajectory)//10)
        
        
        fname_h5 = get_path(self.cfg, filetype="results", overwrite=overwrite)
        with h5py.File(fname_h5, "a") as fh5:
            for i, dist_frame in enumerate(fh5["distances"][self.frame_range[0]:self.frame_range[1]:self.stride]):

                
                distances = dist_frame[mask_pairs]
                
                mask_dist = np.logical_and(r_range[0] < distances, distances < r_range[1]) 

                g_r_frame, edges = np.histogram(distances[mask_dist], range=r_range, bins=n_bins)
                g_r += g_r_frame

                if i%report_stride == 0:
                    print(f"{i:<8d} of {len(self.trajectory):8d}") 


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
        
        if save:
            self._save(key_dict, np.array([r, g_r]), overwrite=overwrite)
        
        if return_all:
            return r, g_r, number_density
    
    def structure_factor_rdf(self,
                             g_r        = None,
                             radii      = None,    
                             tags       = [0,0], 
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
        tags : list of ints
            Particle types for which the rdf should be calculated.
            example tags = [0,1] calculates the rdf between particle types 0 and 1
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
                         save = True,
                         return_all = False,
                         overwrite=True):
        """directly from trajectory"""
        
        # TODO maybe add tags
        
        if not save and not return_all:
            raise ValueError("Either save or return_all has to be True")
        
        key_dict = {"key": "structure_factor",
                    "params": {"qmax": qmax, "n": n}}
        
        # if the calculation was already done, load it
        if self._exists(key_dict):
            fname = get_path(self.cfg, filetype="results", overwrite=overwrite)
            with h5py.File(fname, "a") as fh5:
                key = key_dict["key"]
                Q = fh5[key][0,:]
                S_q = fh5[key][1,:]
                return Q, S_q
            
        n_frames = self.n_frames
        n_atoms = self.cfg.n_particles
        report_stride = int(n_frames//10)
        
        Q = np.linspace(0.0, qmax, n)
        S_q = np.zeros_like(Q)
        S_q[0] = n_atoms**2*n_frames/2
        
        print(f"Calculating structure factor for system {self.cfg.name_sys} ...\n")
        print("Started at ", datetime.datetime.now(), "\n")
        print("Frame    of    Total")
        print("--------------------")
        
        # loop over all distances and calculate the structure factor for all q for each frame
        fname_h5 = get_path(self.cfg, filetype="results", overwrite=True)
        with h5py.File(fname_h5, "a") as fh5:
            
            for k, idx_frame in enumerate(np.arange(self.frame_range[0], self.frame_range[1], self.stride)):
                # load frame one by one and calculate the structure factor
                distances = fh5["distances"][idx_frame][np.triu_indices(n_atoms, k = 1)]
                for n, q in enumerate(Q[1:]):
                    # calc S(q) for every q in current frame
                    S_q[n+1] += np.sum(np.sin(q*distances)/(q*distances))

                if k%report_stride == 0:
                    print(f"{k:<8d} of {n_frames:8d}")                
            print("done\n")
        
        S_q = S_q/n_frames/n_atoms
        
        if save:
            self._save(key_dict, np.array([Q, S_q]), overwrite=overwrite)
        
        if return_all:
            return Q, S_q
    
    def virial_stress_tensor(self,
                             save = True,
                             return_all = False,
                             overwrite = True,
                             particle_type = 0):
        
        key_dict = {"key": "stress_tensor",
                    "params": {"particle_type": particle_type}}
        
        if self._exists(key_dict):
            fname = get_path(self.cfg, filetype="results", overwrite=overwrite)
            with h5py.File(fname, "a") as fh5:
                key = key_dict["key"]
                sigma_t = fh5[key]
                return sigma_t
        
        # sigma = np.zeros((3,3)) # stress tensor reduced units with dimensionality of M/L/T^2
        sigma_t = np.zeros((len(self.trajectory), 3,3)) 
        volume = self.cfg.lbox**3                                                                        # reduced units (volume of box)

        # create mask that selects only one particle type
        # this is important so that a possible tracer in not calculated in the stress tensor
        if particle_type is not None:
            mask_frame = []
            for i in range(len(self.trajectory[0])):
                mask_frame.append(self.topology.tags[i] == self.topology.tags[particle_type])
            mask = np.tile(np.array(mask_frame, dtype=bool), (len(self.trajectory), 3, 1)).transpose(0, 2, 1)
        else:
            mask = np.ones(len(self.trajectory), len(self.trajectory[0]), 3, dtype=bool)
        new_shape = np.array((len(self.trajectory), np.sum(mask[0,:,0]), 3), dtype=int)
        # center box around zero
        traj = self.trajectory[mask].reshape(new_shape) - np.array([self.cfg.lbox/2, self.cfg.lbox/2, self.cfg.lbox/2])

        print("Loading forces ...")
        
        # TODO maybe do this in the setup
        forces_traj = load_results(self.cfg,filetype = ResultsFiletypes.Trajectory.value, frame_range = self.frame_range, stride = self.stride)[mask].reshape(new_shape)
        #forces_traj = load_forces(self.cfg, frame_range=self.frame_range)[mask].reshape(new_shape)
        
        print(f"Calculating stress tensor for system {self.cfg.name_sys} ...\n")
        print("Started at ", datetime.datetime.now(), "\n")
        print("Frame    of    Total")
        print("--------------------")
        
        report_stride = int(len(traj)//10)
        
        # calculate upper triangle of stress tensor
        for i, forces in enumerate(forces_traj):
            for a in range(3):
                for b in range(a, 3):
                    s = np.sum(traj[i,:,a] * forces[:,b])
                    # sigma[a,b] += s
                    sigma_t[i,a,b] += s
            if i%report_stride == 0:
                print(f"{i:<8d} of {len(traj):8d}")

        
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
            self._save(key_dict, sigma_t, overwrite=overwrite)
        
        if return_all:
            return sigma_t
        
    def mean_squared_displacement(self,
                                  tracer_tag = 1,
                                  save = True,
                                  overwrite = False,
                                  return_all = False):
        """
        Calculates the mean squared displacement of the tracer particle
        (averaged over all 3 dimensions)
        
        Parameters
        ----------
        tracer_tag : int
            particle type corresponding to the tracer particle
            
        Returns
        -------
        t : np.ndarray
            time array corresponding to steps in the somulation. (i.e. if stride=10, t = [0, 10, 20, ...])
        msd : np.ndarray
            mean squared displacement of the tracer particle in units of bead radii
        """
        
        key_dict = {"key": "msd",
                    "params": {"tracer_tag": tracer_tag}}
        
        if self._exists(key_dict):
            fname = get_path(self.cfg, filetype="results", overwrite=overwrite)
            with h5py.File(fname, "a") as fh5:
                key = key_dict["key"]
                t = fh5[key][:,0]
                msd = fh5[key][:,1]
                return t, msd
        
        mobility = self.topology.mobility[-1]
        traj_particle = np.squeeze(self.trajectory[:, self.topology.tags == tracer_tag, :])

        print(f"Calculating msd for system {self.cfg.name_sys} ...\n")
        print("Started at ", datetime.datetime.now(), "\n")
        
        # to calculate msd, the pbc shift has to be undone
        frame0 = traj_particle[0]
        traj_abs = np.zeros_like(traj_particle)
        traj_abs[0] = traj_particle[0]
        
        print("Undo pbc shift ...")
        
        for i, frame1 in enumerate(traj_particle[1:]):
            dr = frame0 - frame1
            traj_abs[i+1] = frame1 + self.cfg.lbox * np.round(dr / self.cfg.lbox)
            frame0 = np.copy(traj_abs[i+1])
        
        print("Calculate msd ...")
        
        msd = tid.msd(traj_abs)/3
        
        t = np.arange(len(msd))*self.cfg.stride*self.cfg.timestep
        
        # if save:
        #     fname = get_path(self.cfg, filetype="msd", overwrite=overwrite)
        #     np.save(fname, np.array([t, msd]))
        #     with open(fname[:-4] + ".txt", "w") as f:
        #         f.write(f"msd array with shape [2, {len(msd)}] corresponding to [[t], [msd]]\n")
        #         if self.frame_range is not None:
        #             f.write(f"frame range: {self.frame_range[0]:d} - {self.frame_range[1]:d}\n")
        #         else:
        #             f.write(f"frame range: all\n")
        #         f.write(f"stride: {self.stride:d}\n")
        #         f.write(f"tracer tag: {tracer_tag:d}\n")
        #     print(f"msd saved to {fname}\n")
        
        if save:
            self._save(key_dict, np.array([t, msd]), overwrite=overwrite)
            
        if return_all:
            return t, msd        
    
    def calculate_all(self,
                      save = True,
                      overwrite = False,
                      return_all = False,
                      tracer = False,
                      tracer_tag = 1):
        """
        calculates all properties and saves it into numpy arrays
        """
        print(f"Calculating all properties for system {self.cfg.name_sys} ...\n")

        # r, g_r, number_density = self.rdf(save=save, return_all=True, overwrite=overwrite)
        Q, S_q = self.structure_factor(save=save, return_all=True, overwrite=overwrite)
        sigma_t = self.virial_stress_tensor(save=save, return_all=True, overwrite=overwrite)
        if tracer:
            t_tracer, msd_tracer = self.mean_squared_displacement(tracer_tag=tracer_tag, save=save, return_all=True, overwrite=overwrite)
        print("Finished the calculation of all properties.\n")
        
        if return_all and tracer:
            # return r, g_r, number_density, Q, S_q, sigma_t, t_tracer, msd_tracer
            return Q, S_q, sigma_t, t_tracer, msd_tracer
        elif return_all:
            # return r, g_r, number_density, Q, S_q, sigma_t
            return Q, S_q, sigma_t
    
    def get_timestep_seconds(self,
                             monomer_tag = 0):
        """
        returns the timestep of the current trajectory in seconds
        """
        mu = self.topology.mobility[self.topology.tags[monomer_tag]] # mobility of the system in reduced units
        a = 1e-9*self.cfg.r0_nm                                   # m, reduced legth scale: PEG monomere radius
        r = 1*a                                                      # m, bead radius

        eta_w = 8.53e-4 # Pa*s
        kB = 1.380649e-23 # m^2 kg s^-2 K^-1
        T = 300 # K
        mu_0 = kB*T/(6*np.pi*eta_w*r*a**2) 

        dt_step = mu/mu_0
        # multiply stepwise timestep with the simulation stride and the analysis stride 
        dt = self.cfg.stride*dt_step*self.stride # s
        
        return dt
        
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
        