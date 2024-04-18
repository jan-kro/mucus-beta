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
        
        #! TODO TODO TODO TODO TODO TODO
        # distances do not calculate properly anymore using cell linked lists
        # only the cones within cutoff pbc are calculated!
        
        # TODO THESE MUST ONLY BE DEFINED IF write is set to true respectively
        self.traj_chunk             = None
        self.force_chunk            = None
        self.dist_chunk             = None
        
        # TODO IMPLEMENT THIS PROPERLY
        self.mobility_list          = None
        
        self.cell_index             = None
        self.neighbor_cells_idx     = None
        self.head_array             = None
        self.list_array             = None
        self.n_cells                = None
        self.cell_length            = None
        self.n_neighbor_cells       = None
        
        self.setup()
        
    #! TODO TODO TODO TODO TODO TODO
    # ADD "use forces" TO TOPOLOGY  
      
    def setup(self):
        
        self.n_particles     = self.config.n_particles
        self.box_length      = self.config.lbox
        self.timestep        = self.config.timestep
        self.lB_debye        = self.config.lB_debye # units of beed radii                                       
        
        self.traj_chunk      = np.zeros((self.config.chunksize, self.n_particles, 3))
        self.force_chunk     = np.zeros((self.config.chunksize, self.n_particles, 3))
        self.dist_chunk      = np.zeros((self.config.chunksize, self.n_particles, self.n_particles))
        self.distances       = np.zeros((self.config.chunksize, self.n_particles, self.n_particles))
        
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
        
        # TODO at this stage of the project using pbc is necessary for the simulations to work
        # check for pbc
        if self.config.pbc == True: 
            
            if self.config.cutoff_pbc is None:
                # NOTE here the cutoff of the force with the longest range is used
                cutoff = np.max((self.config.cutoff_debye, self.config.cutoff_LJ))
                # minimal possible cutoff is 1.5*r0
                # otherwise nn calculation breaks
                if cutoff < 4:
                    if self.config.lbox < 8:
                        raise ValueError("Box length is too small for cutoff = 4")
                    cutoff = 4
                self.config.cutoff_pbc = cutoff
        
        # load positions
        self.set_positions(self.topology.positions)
        
        # Define all necessary cell-linked list arrays
        # NOTE the following only works if pbc is used
        
        # calculate the number of cells in each direction
        self.n_cells = int(self.box_length/self.config.cutoff_pbc)
        self.cell_length = self.box_length/self.n_cells
        
        # get the indices of the neighboring cells and self with shape (n_neighbor_cells + 1, 3)
        self.neighbor_cells_idx = np.indices((3, 3, 3), dtype=np.int16).reshape(3, -1).T - 1
        self.n_neighbor_cells = self.neighbor_cells_idx.shape[0]
        
        self.apply_pbc()
        
        self.head_array = -np.ones((self.n_cells, self.n_cells, self.n_cells), dtype=np.int16)
        self.list_array = -np.ones(self.n_particles, dtype=np.int16)
        
        self.update_linked_list()

    def update_linked_list(self):
        """
        updates the list and head array for the current positions
        """
        
        self.list_array.fill(-1)
        self.head_array.fill(-1)
        
        # get cell index for each particle
        self.cell_index = np.floor(self.positions/self.cell_length).astype(np.int16)
        
        for i, cell_idx in enumerate(self.cell_index):
            self.list_array[i] = self.head_array[cell_idx[0], cell_idx[1], cell_idx[2]]
            self.head_array[cell_idx[0], cell_idx[1], cell_idx[2]] = i
    
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
    
    def simulate(self, steps=None):
        """
        Simulates the overdamped langevin dynamics of the system with the defined forcefield using forward Euler.
        """
        
        if steps == None:
            steps = self.config.steps
        
        t_start = time()
        
        n_frames = get_number_of_frames(self.config)
        
        # save initial pos
        if self.config.write_traj==True:
            self.traj_chunk[0] = self.positions
            #fh5_results.create_dataset("trajectory", shape=(n_frames, self.n_particles, 3), dtype="float16")
            fh5_traj = h5py.File(get_path(self.config, filetype='trajectory'), 'w-')
            fh5_traj.create_dataset("trajectory", shape=(n_frames, self.n_particles, 3), dtype="float16")
        
        # define flag for distance writing
        if self.config.write_distances:
            write_distances = True
            fh5_distances = h5py.File(get_path(self.config, filetype='distances'), 'w-')
            fh5_distances.create_dataset("distances", shape=(n_frames, self.n_particles, self.n_particles), dtype="float16")
        else:    
            write_distances = False
        
        rmc.get_forces_cell_linked(
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
                self.dist_chunk[0],
                self.box_length,
                self.config.cutoff_pbc**2,
                self.n_particles,
                3,                          # number of dimensions
                write_distances,
                True,                       # use bond force
                True,                       # use LJ force
                False,                      # use Debye force
                self.neighbor_cells_idx,
                self.head_array,
                self.list_array,
                self.n_cells,
                self.n_neighbor_cells
            )
            
        if self.config.write_forces==True:
            # rmc.get_forces(
            #     self.positions,
            #     self.topology.tags,
            #     self.topology.bond_table,
            #     self.topology.force_constant_nn,
            #     self.topology.r0_bond,
            #     self.topology.sigma_lj,
            #     self.topology.epsilon_lj,
            #     self.topology.q_particle,
            #     self.config.lB_debye,
            #     self.B_debye,
            #     self.forces,
            #     self.dist_chunk[0],
            #     self.box_length,
            #     self.config.cutoff_pbc**2,
            #     self.n_particles,
            #     3,
            #     False,
            #     True,
            #     True,
            #     False
            # )
            
            self.force_chunk[0] = self.forces
            fh5_forces = h5py.File(get_path(self.config, filetype='forces'), 'w-')
            fh5_forces.create_dataset("forces", shape=(n_frames, self.n_particles, 3), dtype="float32")
        
        idx_chunk = 1 # because traj_chunk[0] is already the initial position
        idx_traj = 0
        
        print(f"\nStarting simulation with {steps} steps.")
        for step in tqdm(range(1, steps)):
            
            # calculate forces
            self.forces.fill(0)
            
            # rmc.get_forces(
            #     self.positions,
            #     self.topology.tags,
            #     self.topology.bond_table,
            #     self.topology.force_constant_nn,
            #     self.topology.r0_bond,
            #     self.topology.sigma_lj,
            #     self.topology.epsilon_lj,
            #     self.topology.q_particle,
            #     self.config.lB_debye,
            #     self.B_debye,
            #     self.forces,
            #     self.dist_chunk[idx_chunk],
            #     self.box_length,
            #     self.config.cutoff_pbc**2,
            #     self.n_particles,
            #     3,
            #     write_distances,
            #     True,
            #     True,
            #     False
            # )
            
            self.update_linked_list()
            
            rmc.get_forces_cell_linked(
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
                3,                          # number of dimensions
                write_distances,
                True,                       # use bond force
                True,                       # use LJ force
                False,                      # use Debye force
                self.neighbor_cells_idx,
                self.head_array,
                self.list_array,
                self.n_cells,
                self.n_neighbor_cells
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
                        fh5_traj["trajectory"][idx_traj:idx_traj+self.config.chunksize] = self.traj_chunk
                    
                    if self.config.write_forces:    
                        fh5_forces["forces"][idx_traj:idx_traj+self.config.chunksize] = self.force_chunk
                        
                    if self.config.write_distances:
                        fh5_distances["distances"][idx_traj:idx_traj+self.config.chunksize] = self.dist_chunk
                    
                    idx_traj += self.config.chunksize
                    idx_chunk = 0
                
            self.steps_total += 1
            
        t_end = time()
        
        self.config.simulation_time = t_end - t_start
        
        # fill rest of trajectory in case the steps//stride is not a multiple of the chunksize
        if idx_traj!= n_frames:
            if self.config.write_traj:
                fh5_traj["trajectory"][idx_traj:] = self.traj_chunk[:idx_chunk]
            if self.config.write_forces:
                fh5_forces["forces"][idx_traj:] = self.force_chunk[:idx_chunk]
            if self.config.write_distances:
                fh5_distances["distances"][idx_traj:] = self.dist_chunk[:idx_chunk]
        
        if self.config.write_traj:        
            fh5_traj.close()
        if self.config.write_forces:
            fh5_forces.close()
        if self.config.write_distances:    
            fh5_distances.close()
        
        # save config
        self.config.save()

        return