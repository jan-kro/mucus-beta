import numpy as np
import mdtraj as md
import os
import toml

from .config import Config
from .topology import Topology
from time import time
from pathlib import Path

from copy import deepcopy

class System:
    
    def __init__(self, config: Config):
        
        self.config = config
        self.topology = Topology(self.config)
        
        self.timestep               = None
        self.n_frames               = None
        self.n_beads                = None
        self.B_debye                = None        
        self.box_length             = None
        self.shifts                 = None
        self.positions              = None
        self.forces                 = None
        self.energy                 = None
        self.trajectory             = None
        self.distances              = None
        self.directions             = None
        self.idx_table              = None
        self.idx_interactions       = None
        self.L_nn                   = None
        self.number_density         = None
        
        self.overwrite_switch       = False
        self.version                = ""
        
        self.setup()
    
    # TODO: redo the bond list so that every bond pair only exists once
    
    # TODO make subclasses that 
    #   - handle all the analysis calculations
    #   - handle all the data organisations 
    
    # TODO ceneter box around 0
    
    def setup(self):
        
        self.n_beads         = self.config.n_beads
        self.box_length      = self.config.lbox
        self.timestep        = self.config.timestep
        self.lB_debye        = self.config.lB_debye # units of beed radii                                       
        
        # TODO implement this properly
        self.r0_beeds_nm     = 0.1905 # nm calculated for one PEG Monomer
        self.B_debye         = np.sqrt(self.config.c_S)*self.r0_beeds_nm/10 # from the relationship in the Hansing thesis [c_S] = mM
        self.n_frames        = int(np.ceil(self.config.steps/self.config.stride))
 
        # calculate debye cutoff from salt concentration
        if self.config.cutoff_debye == None:
            self.get_cutoff_debye()
        
        # create nxn index table 
        self.idx_table = np.zeros((2, self.n_beads, self.n_beads), dtype=int)
        for i in range(self.n_beads):
            for j in range(self.n_beads):
                self.idx_table[0, i, j] = i
                self.idx_table[1, i, j] = j
        
        # check for pbc
        if self.config.pbc == True: 
            
            if self.config.cutoff_pbc is None:
                # NOTE here the cutoff of the force with the longest range is used
                cutoff = np.max((self.config.cutoff_debye, self.config.cutoff_LJ))
                # minimal possible cutoff is 1.5*r0
                # otherwise nn calculation breaks
                if cutoff < 3:
                    cutoff = 3
                self.config.cutoff_pbc = cutoff
                   
            self.create_shifts()
        
        # load positions
        self.set_positions(self.topology.positions)
        

    def print_sim_info(self):
        """
        print the config
        """
        
        # this is done so the version is printed but the class variable is not updated
        cfg = deepcopy(self.config)
        cfg.name_sys = self.config.name_sys + self.version
        
        # print everything but bonds
        output = str(cfg).split("bonds")[0]
        output = output.replace(" ", "\n")
        output = output.replace("=", " = ")
        print(output)
        return
    
    def create_shifts(self):
        # array that shifts box for pbc
        self.shifts = np.array(((self.box_length,   0,                0),
                                (0,                 self.box_length,  0),
                                (0,            0,                     self.box_length),
                                (self.box_length,   self.box_length,  0),
                                (self.box_length,   0,                self.box_length),
                                (0,                 self.box_length,  self.box_length),
                                (self.box_length,   self.box_length,  self.box_length),
                                (-self.box_length,  0,                0),
                                (0,                -self.box_length,  0),
                                (0,            0,                    -self.box_length),
                                (-self.box_length,  self.box_length,  0),
                                (-self.box_length,  0,                self.box_length),
                                (0,                -self.box_length,  self.box_length),
                                (self.box_length,  -self.box_length,  0),
                                (self.box_length,   0,               -self.box_length),
                                (0,                 self.box_length, -self.box_length),
                                (-self.box_length, -self.box_length,  0),
                                (-self.box_length,  0,               -self.box_length),
                                (0,                -self.box_length, -self.box_length),
                                (-self.box_length,  self.box_length,  self.box_length),
                                (self.box_length,  -self.box_length,  self.box_length),
                                (self.box_length,   self.box_length, -self.box_length),
                                (-self.box_length, -self.box_length,  self.box_length),
                                (-self.box_length,  self.box_length, -self.box_length),
                                (self.box_length,  -self.box_length, -self.box_length),
                                (-self.box_length, -self.box_length, -self.box_length)))
        return
    
    def set_positions(self, pos):
        
        self.positions = pos
        
        return
    
    def set_cutoff(self, cutoff, ctype = "pbc"):
        
        if ctype.lower() == "pbc":
            self.config.cutoff_pbc = cutoff
        elif ctype.lower() == "lj":
            self.cutoff_LJ = cutoff
        elif ctype.lower() == "debye":
            self.config.cutoff_debye = cutoff
        else:
            raise TypeError(f"Cutoff type \'{ctype:s}\' does not exist.")
        
        return
    
    def get_distances_directions(self):
        """
        updated dist/dir calculation that uses MINIMUM IMAGE CONVENTION, not my bs!
        """
        
        # TODO could be faster by only calculating triu matrix

        # shift box to center
        # TODO DELETE THIS LATER
        
        self.positions -= np.array((self.box_length/2, self.box_length/2, self.box_length/2))
        
        # make 3d verion of meshgrid
        r_left = np.tile(self.positions, (self.n_beads, 1, 1)) # repeats vector along third dimension len(a) times
        r_right = np.reshape(np.repeat(self.positions, self.n_beads, 0), (self.n_beads, self.n_beads, 3)) # does the same but "flipped"

        directions = r_left - r_right # this is right considering the mesh method. dir[i, j] = r_j - r_i

        # apply minimum image convetion
        directions -= self.box_length*np.round(directions/self.box_length)

        # calculate distances and apply interaction cutoff
        distances = np.linalg.norm(directions, axis=2)

        # add cutoff to the self distances so they are not indiced in the intereftions 
        # NOTE this could probably be done in a smarter way
        distances += 2*self.config.cutoff_pbc*np.eye(self.n_beads) # add cutoff to disregard same atoms
        L_box = distances < self.config.cutoff_pbc # NOTE the "<" is important, because if it was "<=" the diagonal values would be included
        
        # only save dist/dir, that lie within the interaction cutoff
        self.directions = directions[L_box]
        self.distances  = distances[L_box]
        self.idx_interactions = self.idx_table[:, L_box].T

        # shift box back again
        # TODO delete later
        self.positions += np.array((self.box_length/2, self.box_length/2, self.box_length/2))
        
        # loop through index list and see if any tuple corresponds to bond        
        L_nn = list(())
        for idx in self.idx_interactions:
            if np.any(np.logical_and(idx[0] == self.topology.bonds[:, 0], idx[1] == self.topology.bonds[:, 1])):
                L_nn.append(True)
            else:
                L_nn.append(False)
                
        self.L_nn = L_nn
        
        return

    def get_forces(self):
        """
        Delete all old forces and add all forces occuring in resulting from the self.positions configuration
        
        This only works because the forces are defined in a way where they are directly added to the self.forces variable
        """
        
        # delete old forces
        self.forces = np.zeros((self.n_beads, 3))
        
        # add new forces
        self.force_NearestNeighbours()
        self.force_LennardJones_cutoff()
        self.force_Debye()
        
        return
    
    def get_forces_test(self, output=False):
        """
        for testing the forces
        """
        
        print("Position")
        print(self.positions)
        
        self.get_distances_directions()
        
        #delete old forces
        self.forces = np.zeros((self.n_beads, 3))
        self.force_NearestNeighbours()
        print("nearest neighbours")
        print(self.forces)
        
        force_nn = deepcopy(self.forces)
        
        self.forces = np.zeros((self.n_beads, 3))
        self.force_LennardJones_cutoff()
        print("Lennard Jones")
        print(self.forces)
        
        force_lj = deepcopy(self.forces)
        
        self.forces = np.zeros((self.n_beads, 3))
        self.force_Debye()
        print("Debye")
        print(self.forces)
        
        force_deb = deepcopy(self.forces)
        
        if output == True:
            return force_nn, force_lj, force_deb

    
    def force_NearestNeighbours(self):
        """
        harmonice nearest neighhbour interactions
        """
                
        idxs = self.idx_interactions[self.L_nn]
        distances = self.distances[self.L_nn].reshape(-1,1)
        directions = self.directions[self.L_nn]
        force_constants = self.topology.force_constant_nn[self.topology.get_tags(idxs[:,0], idxs[:,1])].reshape(-1,1)
        
        # calculate the force of every bond at once
        forces_temp = force_constants*(2-4/distances)*directions # NOTE r0 is hardcoded with r0=2

        for i, force in zip(idxs[:, 0], forces_temp):
            self.forces[i, :] += force
        
        return

    def force_LennardJones_cutoff(self):
        """
        LJ interactions using a cutoff
        """
        
        L_lj = self.distances < self.config.cutoff_LJ
        
        idxs = self.idx_interactions[L_lj]
        distances = self.distances[L_lj].reshape(-1, 1)
        directions = self.directions[L_lj]
        epsilon = self.topology.epsilon_lj[self.topology.get_tags(idxs[:,0], idxs[:,1])].reshape(-1,1)
        sigma = self.topology.sigma_lj[self.topology.get_tags(idxs[:,0], idxs[:,1])].reshape(-1,1)
        
        forces_temp = 4*epsilon*(-12*sigma**12/distances**14 + 6*sigma**7/distances**8)*directions

        for i, force in zip(idxs[:, 0], forces_temp):
            self.forces[i, :] += force
            
        return

    def force_Debye(self):
        """
        non bonded interaction (debye screening)
        """
        
        # exclude bonds
        L_nb = np.logical_not(self.L_nn)
        
        idxs = self.idx_interactions[L_nb]
        distances = self.distances[L_nb].reshape(-1, 1)
        directions = self.directions[L_nb]
        q2 = self.topology.q_bead[idxs[:,0]]*self.topology.q_bead[idxs[:,1]]
        
        # since the debye cutoff is used for the dist/dir cutoff the distances dont have to be checked
        forces_temp = -q2*self.config.lB_debye*(1+self.B_debye*distances)*np.exp(-self.B_debye*distances)*directions/distances**3
        
        for i, force in zip(idxs[:, 0], forces_temp):
            self.forces[i, :] += force
        
        return

    def get_cutoff_debye(self, eps=1):
        """
        use the maximum charge in the config to determine the debey force cutoff
        
        the distance, where the force is smaller than the treshold eps is the debey cutoff
        """
        r = 0
        dr = 0.05
        force = 9999 
        while np.max(force) > eps:
            r += dr
            force = np.max(self.topology.q_bead)**2*self.config.lB_debye*(1+self.B_debye*r)*np.exp(-self.B_debye*r)/r**2
            
        self.config.cutoff_debye = r
        
        return 
    
    def force_Random(self):
        """
        Gaussian random Force with a per particle standard deviation of sqrt(6 mu_0) w.r.t. its absolute value
        """
        
        # since the std of the foce should be sqrt(6*mu) but the std of the absolute randn vector is sqrt(3)
        # the std used here is sqrt(2*mu)
        
        return np.sqrt(2*self.config.timestep*self.topology.mobility)*np.random.randn(self.n_beads, 3)
    
    def rdf(self,
            r_range = None,
            n_bins = None,
            bin_width = 0.05,
            save=True,
            overwrite=False):
        """
        DEPRECATED
        """
        
        if r_range is None:
            r_range = np.array([1.5, self.box_length/2])
            
        if n_bins is None:   
            n_bins = int((r_range[1] - r_range[0]) / bin_width)
        
        fname_top = self.create_fname(filetype="topology")

        if not os.path.exists(fname_top):
            self.create_topology_pdb()


        natoms = self.n_beads
        lbox = self.box_length

        # create unic cell information
        uc_vectors = np.repeat([np.array((lbox, lbox, lbox))], len(self.trajectory), axis=0)
        uc_angles = np.repeat([np.array((90,90,90))], len(self.trajectory), axis=0)

        # create mdtraj trajectory object
        trajectory = md.Trajectory(self.trajectory, md.load(fname_top).topology, unitcell_lengths=uc_vectors, unitcell_angles=uc_angles)
        
        # create bond pair list
        pairs = list()
        for i in range(natoms-1):
            for j in range(i+1, natoms):
                pairs.append((i, j))
                
        pairs = np.array(pairs)
        
        self.number_density = len(pairs) * np.sum(1.0 / trajectory.unitcell_volumes) / natoms / len(trajectory)

        r, gr = md.compute_rdf(trajectory, pairs, r_range=(1.5, 20), bin_width=0.05)
        
        if save == True:
            fname = self.create_fname(filetype="rdf", overwrite=overwrite)
            np.save(fname, np.array([r, gr]))
        
        return r, gr
    
    def get_structure_factor_rdf(self,
                                 radii      = None,
                                 g_r        = None,     
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
            radii, g_r = self.rdf(overwrite=overwrite)
            
        if rho is None:
            rho = self.number_density
        
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
            fname = self.create_fname(filetype="structure_factor", overwrite=overwrite)
            np.save(fname, np.array([Q, S_q]))
        
        return Q, S_q
    
    
    def create_fname(self, 
                     filetype: str = "trajectory", 
                     overwrite: bool = False):
        """
        Creates an outfile str for a specified filetype. If the directory does not exist, it is created.
        The overwrite call checks if the system name already exists in the output directory. If it does, 
        a version str ("_vXX") is appended to the system name. 
        
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
            if self.overwrite_switch == False:
                self.overwrite_switch = True # this prevents the version from update everytime create_fname() is called 
                k_max = 0
                # check if any file already exists
                for rel_path, ftype in dir_dict.values():
                    
                    # do not check initialisation files, since they must exist beforhand
                    if ftype == ".toml":
                        continue
                    if rel_path == "/initial_positions/xyz_":
                        continue
                    
                    fname = self.config.dir_sys + rel_path + self.config.name_sys + ftype
                    k = 0
                    while os.path.exists(fname):
                        self.version = f"_v{k:d}" # update version until no files are found
                        fname = self.config.dir_sys + rel_path + self.config.name_sys + self.version + ftype
                        if k > k_max:
                            k_max = deepcopy(k)
                        k += 1
                if k_max > 0:
                    self.version = f"_v{k_max:d}" # change version to kmax
            
            version = self.version
        else:
            version = ""
        
        # create outfile string
        fname = self.config.dir_sys + dir_dict[filetype][0] + self.config.name_sys + version + dir_dict[filetype][1]
        
        # make dir if path doesnt exist
        path = Path(fname).parent
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        # TODO implement this instead of the above
        # head_tail = os.path.split(fname)
        # base = head_tail[0]
        # name_tot, ext = os.path.splitext(head_tail[1])
        # split_arg = "_v"

        # name_version = name_tot.split(split_arg)
        # if len(name_version) == 1:
        #     name = name_version[0]
        #     version = 0
        # else:
        #     try:
        #         # make sure that version is an integer, otherwise it might just be a random "_vABC" string (idk eg. xyz_variable_input.npy)
        #         version = int(name_version[-1])
        #         name = name_version[0]
        #         # reassemble name in case there is a second "_v" in the name
        #         for part in name_version[1:-1]:
        #             name += split_arg + part
        #     except:
        #         # use total name as name
        #         name = name_tot
        #         version = 0

        # print(base, name, version, ext)

        # if overwrite == False:
        #     version += 1
        #     while os.path.exists(base + "/" + name + split_arg + str(version) + ext):
        #         version += 1

        # if version > 0:
        #     fname = base + "/" + name + split_arg + str(version) + ext
        
        return fname
    
    def write_frame_gro(self, n_atoms, coordinates, time, fname, comment="", box=None, precision=3):

        f = open(fname, "a")

        comment += ', t= %s' % time

        varwidth = precision + 5
        fmt = '%%5d%%-5s%%5s%%5d%%%d.%df%%%d.%df%%%d.%df' % (
                varwidth, precision, varwidth, precision, varwidth, precision)

        lines = [comment, ' %d' % n_atoms]
        # if box is None:
        #     box = np.zeros((3,3))

        for i in range(n_atoms):
            lines.append(fmt % (i+1, "HET", "CA", i+1,
                                coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]))
            # lines.append(fmt % (residue.resSeq, residue.name, atom.name, serial,
            #                     coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]))

        # lines.append('%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f' % (
        #     box[0,0], box[1,1], box[2,2],
        #     box[0,1], box[0,2], box[1,0],
        #     box[1,2], box[2,0], box[2,1]))

        lines.append('%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f' % (0,0,0,0,0,0,0,0,0))
        
        f.write('\n'.join(lines))
        f.write('\n')
        f.close()
    
    def load_traj_ndarray(self, traj):
        
        self.trajectory = traj
        self.positions = traj[-1]
        
        return
    
    def load_traj_gro(self, 
                      fname: str = None,
                      overwrite: bool = False):
         
        if fname is None:
            fname = self.create_fname(filetype="trajectory", overwrite=overwrite)
        if os.path.exists(fname) == False:
            raise TypeError(f"Error: file \"{fname:s}\" does not exist")
        
        if overwrite == False:
            # TODO remove the "or" once the initialisation is fixed
            if self.trajectory is None:
                self.trajectory = md.load(fname).xyz
            else:
                raise Exception("Error: System already has a trajectory")
        else:
            self.trajectory = md.load(fname).xyz
        
        
        self.positions = self.trajectory[-1]
        
    def save_config(self, overwrite=False):
        """
        saves current self.config in a .toml file 
        """
        
        fname_sys = self.create_fname(filetype="config", overwrite=overwrite)
        
        # NOTE this should be reset after the file is saved 
        # TODO redo the versions in another way  
        if overwrite == False:
            self.config.name_sys += self.version
        
        output = str(self.config)
        output = output.replace("=True", "=true") # for toml formating
        output = output.replace("=False", "=false")
        output = output.replace(" ", "\n")
        output = output.replace("=", " = ")

        f = open(fname_sys, "w")
        f.write(output)
        f.close()
        
        # remove version again from name_sys string
        # only in case the save_config method (or any other using self.config.name_sys) is called again
        if overwrite == False:
            self.config.name_sys = self.config.name_sys[:-len(self.version)]
        
        return
    
    
    def apply_pbc(self):
        """
        Repositions all atoms outside of the box to the other end of the box
        """
        
        # calculate modulo L
        self.positions = np.mod(self.positions, self.box_length)
        
        return
    
    def simulate(self, steps=None, save_sys=True, overwrite_sys=False, report_time=True, report_stride=1000):
        """
        Simulates the brownian motion of the system with the defined forcefield using forward Euler.
        """
        
        # NOTE This doeasnt make sense enymore
        # if self.positions is None:
        #     self.create_chain()
        
        if steps == None:
            steps = self.config.steps
        
        t_start = time()
        
        idx_traj = 1 # because traj[0] is already the initial position
        
        fname_traj = self.create_fname(filetype="trajectory", overwrite=overwrite_sys)
        self.write_frame_gro(self.n_beads, self.positions, 0.0, fname_traj, comment=f"traj step 0") # write frame 0 with initial positions
        
        for step in range(1, steps):
            
            # get distances for interactions
            self.get_distances_directions()
            
            # get forces
            self.get_forces()
            
            # integrate
            self.positions = self.positions + self.config.timestep*self.topology.mobility*self.forces + self.force_Random()
            
            # apply periodic boundary conditions
            self.apply_pbc()

            
            if (self.config.write_traj==True) and (step%self.config.stride==0):
                    
                # TODO add condition for direct writing
                self.write_frame_gro(self.n_beads, self.positions, float(idx_traj), fname_traj, comment=f"traj step {step:d}")
                idx_traj += 1
            
            if (report_time==True) and (step%report_stride==0):
                print(f"Step {step:12d} of {steps:12d} | {int((time()-t_start)//60):6d} min {int((time()-t_start)%60):2d} s")
            
            # if np.any(self.distances[self.L_nn] > 5):
            #     print("System exploded")
            #     print("simulation Step", step)
            #     break
            
        t_end = time()
        
        self.config.simulation_time = t_end - t_start
        
        if save_sys == True:
            self.save_config(overwrite=overwrite_sys)

        return