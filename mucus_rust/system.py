import numpy as np
import os
import toml
import h5py

import rust_mucus as rmc

from .config import Config
from .topology import Topology
from .utils import get_path, get_number_of_frames
from pathlib import Path
from tqdm import tqdm

from time import time
from copy import deepcopy

class System:
    
    def __init__(self, config: Config):

        self.config = config
        self.topology = Topology(self.config)
        
        self.timestep               = None
        self.n_frames               = None
        self.chunksize              = None
        self.n_particles            = None
        self.B_debye                = None        
        self.box_length             = None
        self.shifts                 = None
        self.positions              = None
        self.forces                 = None
        self.energy                 = None
        self.distances              = None
        self.directions             = None
        self.idx_table              = None
        self.idx_interactions       = None
        self.L_nn                   = None
        self.number_density         = None
        
        self.overwrite_switch       = False
        self.version                = ""
        self.steps_total            = 0
        
        self.traj_chunk             = None
        self.force_chunk            = None
        self.dist_chunk             = None
        
        # TODO IMPLEMENT THIS PROPERLY
        self.mobility_list          = None
        self.bonds                  = None
        
        self.setup()
        
    # THIS !!!!!
    # TODO ceneter box around 0
    
    #TODO
    #! first distances frame is appearently all zeros
    
    def setup(self):
        
        self.n_particles     = self.config.n_particles
        self.box_length      = self.config.lbox
        self.timestep        = self.config.timestep
        self.lB_debye        = self.config.lB_debye # units of beed radii                                       
        
        self.traj_chunk      = np.zeros((self.config.chunksize, self.n_particles, 3))
        self.force_chunk     = np.zeros((self.config.chunksize, self.n_particles, 3))
        self.dist_chunk      = np.zeros((self.config.chunksize, self.n_particles, self.n_particles))
        
        self.forces          = np.zeros((self.n_particles, 3), dtype=np.float64)
        
        # TODO implement this properly
        self.r0_beeds_nm     = self.config.r0_nm # 0.1905 # nm calculated radius for one PEG Monomer
        self.B_debye         = np.sqrt(self.config.c_S)*self.r0_beeds_nm/10 # from the relationship in the Hansing thesis [c_S] = mM
        self.n_frames        = int(np.ceil(self.config.steps/self.config.stride))
        self.mobility_list   = np.array([self.topology.mobility[i] for i in self.topology.tags], ndmin=2).reshape(-1, 1)
        
        
        
        # calculate debye cutoff from salt concentration
        if self.config.cutoff_debye == None:
            self.get_cutoff_debye()
        
        # create nxn index table 
        self.idx_table = np.zeros((2, self.n_particles, self.n_particles), dtype=int)
        for i in range(self.n_particles):
            for j in range(self.n_particles):
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
        
        # create bond_list
        self.bonds = []
        for i in range(self.n_particles-1):
            for j in range(i+1, self.n_particles):
                if self.topology.bond_table[i, j] == True:
                    self.bonds.append((i, j))
        self.bonds = np.array(self.bonds)
        
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
    
    def set_timestep(self, timestep):
        
        self.timestep = timestep
        
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
    
    # TODO MOVE THIS TO UTILS
    def get_cutoff_debye(self, eps=1):
        """
        use the maximum charge in the config to determine the debey force cutoff
        
        the distance, where the force is smaller than the treshold eps is the debey cutoff
        """
        r = 0
        dr = 0.05
        force = 9999
        # TODO CHANGE TO MAX DISPLACEMENT = 0.1 
        while np.max(force) > eps:
            r += dr
            force = np.max(self.topology.q_particle)**2*self.config.lB_debye*(1+self.B_debye*r)*np.exp(-self.B_debye*r)/r**2
            
        self.config.cutoff_debye = r
        
        return 
    
    def force_Random(self):
        """
        Gaussian random Force with a per particle standard deviation of sqrt(6 mu_0) w.r.t. its absolute value
        """
        
        # since the std of the foce should be sqrt(6*mu) but the std of the absolute randn vector is sqrt(3)
        # the std used here is sqrt(2*mu)
        
        # TODO THIS WILL BREAK WHEN MOBILITY IS CHANGED TO SHAPE (ntags)
        return np.sqrt(2*self.timestep*self.mobility_list)*np.random.randn(self.n_particles, 3)    
    
    
    # TODO USE utils.get_fname INSTEAD
    
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
            "forces"
        """
        
        # if the system should not be overwritten change the version
        dir_dict = {"trajectory":       ("/trajectories/traj_",                 ".gro"),
                    "config":           ("/configs/cfg_",                       ".toml"),
                    "parameters":       ("/parameters/param_",                  ".toml"),
                    "init_pos":         ("/initial_positions/xyz_",             ".npy"),
                    "rdf":              ("/results/rdf/rdf_",                   ".npy"),
                    "structure_factor": ("/results/structure_factor/Sq_",       ".npy"),
                    "forces":           ("/forces/forces_",                     ".txt")}
        
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
    
    def write_frame_gro(self, n_atoms, coordinates, time, file, comment="", box=None, precision=3):

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
        
        file.write('\n'.join(lines))
        file.write('\n')
        
    def write_frame_force(self, frame, file, frame_number):
        file.write(f"Frame {frame_number}\n")
        for line in frame:
            file.write(f"{line[0]:.9f} {line[1]:.9f} {line[2]:.9f}\n")
        
    def save_config(self, overwrite=False, print_fname=False):
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
            
        if print_fname == True:
            print(fname_sys)
        
        return
    
    
    def apply_pbc(self):
        """
        Repositions all atoms outside of the box to the other end of the box
        """
        
        # calculate modulo L
        self.positions = np.mod(self.positions, self.box_length)
        
        return
    
    def simulate(self, steps=None, max_dist=None, check_explosion=True):
        """
        Simulates the brownian motion of the system with the defined forcefield using forward Euler.
        """
        
        if steps == None:
            steps = self.config.steps
            
        if max_dist == None:
            max_dist = self.config.cutoff_pbc
        
        t_start = time()
        
        idx_chunk = 1 # because traj_chunk[0] is already the initial position
        idx_traj = 0
        
        # initialize results h5 file
        fh5_results = h5py.File(get_path(self.config, filetype='results'), 'w-')
        
        n_frames = get_number_of_frames(self.config)
        
        # save initial pos
        if self.config.write_traj==True:
            self.traj_chunk[0] = self.positions
            fh5_results.create_dataset("trajectory", shape=(n_frames, self.n_particles, 3), dtype="float64")
            
        if self.config.write_forces==True:
            rmc.get_forces(
                self.positions,
                self.topology.tags,
                self.topology.bond_table,
                self.topology.force_constant_nn,
                self.topology.r0_bond,
                self.topology.sigma_lj,
                self.topology.epsilon_lj,
                self.topology.q_particle,
                self.config.lB_debye,
                self.B_debye,
                self.forces,
                self.dist_chunk[idx_chunk],
                self.box_length,
                self.config.cutoff_pbc**2,
                self.n_particles,
                3,
                False,
                True,
                True,
                False
            )
            
            self.force_chunk[0] = self.forces
            fh5_results.create_dataset("forces", shape=(n_frames, self.n_particles, 3), dtype="float64")
        
        # define flag for distance writing
        if self.config.write_distances:
            write_distances = True
            fh5_results.create_dataset("distances", shape=(n_frames, self.n_particles, self.n_particles), dtype="float64")
        else:    
            write_distances = False
        
        print(f"\nStarting simulation with {steps} steps.")
        for step in tqdm(range(1, steps)):
            
            # calculate forces
            self.forces.fill(0)
            rmc.get_forces(
                self.positions,
                self.topology.tags,
                self.topology.bond_table,
                self.topology.force_constant_nn,
                self.topology.r0_bond,
                self.topology.sigma_lj,
                self.topology.epsilon_lj,
                self.topology.q_particle,
                self.config.lB_debye,
                self.B_debye,
                self.forces,
                self.dist_chunk[idx_chunk],
                self.box_length,
                self.config.cutoff_pbc**2,
                self.n_particles,
                3,
                write_distances,
                True,
                True,
                False
            )
            
            # reset distance flag until next stride
            write_distances = False
            
            # integrate                                     # TODO implement mobility in forces
            self.positions = self.positions + self.timestep*self.mobility_list*self.forces + self.force_Random()
            
            # apply periodic boundary conditions (0, L) x (0, L) x (0, L)
            self.apply_pbc()

            
            if step%self.config.stride==0:
                
                if self.config.write_traj:    
                    self.traj_chunk[idx_chunk] = self.positions
                
                if self.config.write_forces:
                    self.force_chunk[idx_chunk] = self.forces
                    
                if self.config.write_distances:
                    write_distances = True
                    
                idx_chunk += 1
                
                if idx_chunk == self.config.chunksize:
                    if self.config.write_traj:
                        fh5_results["trajectory"][idx_traj:idx_traj+self.config.chunksize] = self.traj_chunk
                    
                    if self.config.write_forces:    
                        fh5_results["forces"][idx_traj:idx_traj+self.config.chunksize] = self.force_chunk
                        
                    if self.config.write_distances:
                        fh5_results["distances"][idx_traj:idx_traj+self.config.chunksize] = self.dist_chunk
                    
                    idx_traj += self.config.chunksize
                    idx_chunk = 0
                
                    # if check_explosion: 
                    #     #! THIS TYPE OF INDEXING DOESNT WORK ANYMORE
                    #     d = self.positions[self.bonds[:, 0]] - self.positions[self.bonds[:, 1]]
                    #     d -= np.round(d/self.box_length)*self.box_length
                    #     if np.any(np.abs(d) > max_dist):
                    #         print(f"Distances between bonded atoms larger than maxdist = {max_dist}")
                    #         print("System exploded")
                    #         print("simulation Step", step)
                    #         break
                
            self.steps_total += 1
            
        t_end = time()
        
        self.config.simulation_time = t_end - t_start
        
        # fill rest of trajectory in case the steps//stride is not a multiple of the chunksize
        if idx_traj!= n_frames:
            if self.config.write_traj:
                fh5_results["trajectory"][idx_traj:] = self.traj_chunk[:idx_chunk-1]
            if self.config.write_forces:
                fh5_results["forces"][idx_traj:] = self.force_chunk[:idx_chunk-1]
                
        fh5_results.close()
        
        # save config
        self.config.save()

        return