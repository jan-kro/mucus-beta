import os
import toml
import numpy as np
from typing import Optional
from mucus.config import Config
from pathlib import Path


def get_path(Config: Config, 
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
        """
        
        # if the system should not be overwritten change the version
        dir_dict = {"trajectory":       ("/trajectories/traj_",                 ".gro"),
                    "config":           ("/configs/cfg_",                       ".toml"),
                    "parameters":       ("/parameters/param_",                  ".toml"),
                    "init_pos":         ("/initial_positions/xyz_",             ".npy"),
                    "rdf":              ("/results/rdf/rdf_",                   ".npy"),
                    "structure_factor": ("/results/structure_factor/Sq_",       ".npy")}
        
        if overwrite == False:
            # check if any file already exists
            for rel_path, ftype in dir_dict.values():
                fname = Config.dir_sys + rel_path + Config.name_sys + ftype
                k = 0
                while os.path.exists(fname):
                    version = f"_v{k:d}" # update version until no files are found
                    fname = Config.dir_sys + rel_path + Config.name_sys + version + ftype
                    k += 1
                if k > 0:
                    version = f"_v{k-1:d}"
                else:
                    version = ""
        else:
            version = ""
        
        # create outfile string
        fname = Config.dir_sys + dir_dict[filetype][0] + Config.name_sys + version + dir_dict[filetype][1]
        
        # make dir if path doesnt exist
        path = Path(fname).parent
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        return fname
    
def get_version(config: Config):
    """
    returns an integer of the name_sys version string. If no version is found, 0 is returned.
    """

    split_arg = "_v"
    name_version = config.name_sys.split(split_arg)
    if len(name_version) == 1:
        version = 0
    else:
        try:
            # make sure that version is an integer, otherwise it might just be a random "_vABC" string (idk eg. xyz_variable_input.npy)
            version = int(name_version[-1])
        except:
            version = 0

    return version

    
def get_rdf(config: Optional[Config],
            r_range: tuple = None,
            n_bins : int = None,
            bin_width: tuple = 0.05,
            tag1: int = 0,
            tag2: int = 0,
            save: bool =True):
    """
    Calculates the radial distribution function for a given trajectory specified in the config file.
    
    Parameters:
        config: Config object
            Containing all information of the system to analyze
        r_range: tuple of length 2 containing the minimum and maximum radius
        n_bins: number of bins
        bin_width: width of the bins
        tag1: particle tag 1
        tag1: particle tag 2
        save: if True, the rdf is saved in the results folder
        
    Returns:
        r: array of length n_bins containing the radii of the bins
        gr: array of length n_bins containing the rdf values
    """
    
    # TODO implement particle type dependency
    
    # if config is given as a path, load it
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    # check if trajectory exists
    fname_traj = get_path(config, filetype="trajectory")
    if not os.path.exists(fname_traj):
        raise FileNotFoundError(f"trajectory file {fname_traj} not found")
    
    n_atoms = config.n_beads
    l_box = config.lbox
    n_frames = int(np.ceil(config.steps/config.stride))
    
    if r_range is None:
        r_range = np.array([1.5, l_box/2])
    
    if n_bins is None:
        n_bins = int((r_range[1] - r_range[0]) / bin_width)
    
    # load particle tags
    fname_top = get_path(config, filetype="topology")
    params = toml.load(open(fname_top, encoding="UTF-8"))
    tags = np.array(params["tags"])
            
    # create bond pair list
    pairs = list()
    for i in range(n_atoms-1):
        for j in range(i+1, n_atoms):
            # check if tags match
            if tags[i] == tag1 and tags[j] == tag2:
                pairs.append((i, j))

def load_trajectory(config: Optional[Config],
                    frame_range: tuple = None):
    """
    Loads Trajectory into numpy array forthe given range of frames.
    If frame_range is None, the whole trajectory is loaded.
    """
    from io import StringIO
    
    # if config is given as a path, load it
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    n_frames = int(np.ceil(config.steps/config.stride))
    if frame_range is None:
        frame_range = (0, n_frames)
    elif frame_range[0] < 0 or frame_range[1] > n_frames:
            raise ValueError("frame range out of bounds")
    
    #define a np.dtype for gro array/dataset (hard-coded for now)
    gro_dt = np.dtype([('col1', 'S4'), ('col2', 'S4'), ('col3', int), 
                    ('col4', float), ('col5', float), ('col6', float)])

    # initialize trajectory array
    traj = np.zeros((frame_range[1] - frame_range[0], config.n_beads, 3))
    idx_traj = 0
    
    fname_traj = get_path(config, filetype="trajectory")
    with open(fname_traj, "r") as f:
        
        for idx_frame in range(n_frames):
            if idx_frame == frame_range[1]:
                break
            if frame_range[0] <= idx_frame < frame_range[1]: 
                # skip header
                for i in range(2):
                    f.readline()
                # save data into string
                data_str = ""
                for i in range(config.n_beads):
                    data_str += f.readline()        
                # skip box size
                f.readline()
                # read in data
                frame = np.genfromtxt(StringIO(data_str), dtype=gro_dt, usecols=(3, 4, 5))
                traj[idx_traj] = np.array(frame.tolist())
                idx_traj += 1
            else:
                # skip frame
                for i in range(config.n_beads + 3):
                    f.readline()
    f.close()
            
    return traj

def get_distances(config: Optional[Config],
                  frame = None,
                  range: tuple = None):
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    if frame is None:
        frame = np.load(get_path(config, filetype="init_pos"))
    if range is None:
        range = (0, config.lbox/2)
    if range[0] < 0 or range[1] > config.lbox/2:
        raise ValueError("range out of bounds")
    
    # make 3d verion of meshgrid
    r_left = np.tile(frame, (config.n_beads, 1, 1)) # repeats vector along third dimension len(a) times
    r_right = np.reshape(np.repeat(frame, config.n_beads, 0), (config.n_beads, config.n_beads, 3)) # does the same but "flipped"

    directions = r_left - r_right # this is right considering the mesh method. dir[i, j] = r_j - r_i

    # apply minimum image convetion
    directions -= config.lbox*np.round(directions/config.lbox)

    # calculate distances and apply interaction cutoff
    distances = np.linalg.norm(directions, axis=2)

    # apply cutoff
    mask = distances < config.cutoff_pbc
    
    # only return distances, that lie within the interaction cutoff
    return distances[mask]     
    
    
    
# def rdf(self,
#             r_range = None,
#             n_bins = None,
#             bin_width = 0.05,
#             save=True,
#             overwrite=False):
        
#         if r_range is None:
#             r_range = np.array([1.5, config.lbox/2])
            
#         if n_bins is None:   
#             n_bins = int((r_range[1] - r_range[0]) / bin_width)
        
#         fname_top = self.create_fname(filetype="topology")

#         if not os.path.exists(fname_top):
#             self.create_topology_pdb()


#         natoms = len(self.trajectory[0])
#         lbox = config.lbox

#         # create unic cell information
#         uc_vectors = np.repeat([np.array((lbox, lbox, lbox))], len(self.trajectory), axis=0)
#         uc_angles = np.repeat([np.array((90,90,90))], len(self.trajectory), axis=0)

#         # create mdtraj trajectory object
#         trajectory = md.Trajectory(self.trajectory, md.load(fname_top).topology, unitcell_lengths=uc_vectors, unitcell_angles=uc_angles)
        
#         # create bond pair list
#         pairs = list()
#         for i in range(natoms-1):
#             for j in range(i+1, natoms):
#                 pairs.append((i, j))
                
#         pairs = np.array(pairs)
        
#         self.number_density = len(pairs) * np.sum(1.0 / trajectory.unitcell_volumes) / natoms / len(trajectory)

#         r, gr = md.compute_rdf(trajectory, pairs, r_range=(1.5, 20), bin_width=0.05)
        
#         if save == True:
#             fname = self.create_fname(filetype="rdf", overwrite=overwrite)
#             np.save(fname, np.array([r, gr]))
        
#         return r, gr