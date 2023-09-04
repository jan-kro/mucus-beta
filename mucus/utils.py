import os
import toml
import numpy as np
from .config import Config
from .system import System
from typing import Optional
from pathlib import Path
from io import StringIO
from tqdm import tqdm


def get_path(Config: Optional[Config], 
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
    
    
    dir_dict = {"trajectory":           ("/trajectories/traj_",                 ".gro"),
                "config":               ("/configs/cfg_",                       ".toml"),
                "parameters":           ("/parameters/param_",                  ".toml"),
                "init_pos":             ("/initial_positions/xyz_",             ".npy"),
                "rdf":                  ("/results/rdf/rdf_",                   ".npy"),
                "structure_factor":     ("/results/structure_factor/Sq_",       ".npy"),
                "structure_factor_rdf": ("/results/structure_factor/Sq_rdf_",       ".npy"),
                "stress_tensor":        ("/results/stress_tensor/sigma_",       ".npy"),
                "forces":               ("/forces/forces_",                     ".txt")}
    
    if isinstance(Config, str):
        fname = Config.split("/configs/cfg_")[0] + dir_dict[filetype][0] + Config.split("/configs/cfg_")[1].split(".")[0] + dir_dict[filetype][1]
    else:
        fname = Config.dir_sys + dir_dict[filetype][0] + Config.name_sys + dir_dict[filetype][1]
    
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
            # if file exist update version until no vile with that version exists
            while os.path.exists(base + "/" + name + split_arg + str(version) + ext):
                version += 1
            # update fname with new version
            fname = base + "/" + name + split_arg + str(version) + ext
    
    # do not change version for init pos and parameters (DEPRECATED)
    # the following should deal with backwards compatability (not fully fixable)
    # in the case of overwrite it will get the old file if the current version doesnt exist
    # NOTE THIS IS STUPID AND RUNS INTO PROBLEMS SO THERE IS NO BACKWARDS COMPATABILITY FUCK IT
    # if (filetype == "init_pos" or filetype == "parameters") and (overwrite == True):
    #     if not os.path.exists(fname):
    #         fname = base + "/" + name + ext

    # if parent path doesnt exist, create it
    if not os.path.exists(Path(fname).parent):
        os.makedirs(Path(fname).parent)
            
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

def load_forces(config: Optional[Config],
                frame_range: list = None,
                stride: int = 1):
    """
    Loads Forces into numpy array for the given range of frames.
    If frame_range is None, all frames are loaded.
    """
    
    # if config is given as a path, load it
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    # get traj path
    fname_traj = get_path(config, filetype="forces")
    
    if not os.path.exists(fname_traj):
        print(f"forces file {fname_traj} not found")
        print(f"Creating forces file from {config.name_sys}.gro ...")
        
        # create sys and load trajectory
        traj = load_trajectory(config)
        sys = System(config)
        
        f_force = open(fname_traj, "a")
        
        for i, frame in tqdm(enumerate(traj)):
            sys.set_positions(frame)
            sys.get_distances_directions()
            sys.get_forces()
            sys.write_frame_force(sys.forces, f_force, i)

        f_force.close()
        print("Done.")
    
    # get number of frames
    n_frames = get_number_of_frames(config)
    
    # handle frame range input
    if frame_range is None:
        frame_range = (0, n_frames)
    if frame_range[0] == None:
        frame_range[0] = 0
    if frame_range[1] == None:
        frame_range[1] = n_frames
    if frame_range[0] < 0 or frame_range[1] > n_frames:
        raise ValueError("frame range out of bounds")

    # initialize forces array
    forces = np.zeros(((frame_range[1] - frame_range[0] - 1)//stride + 1, config.n_beads, 3))
    idx_traj = 0
    
    with open(fname_traj, "r") as f:
        
        for idx_frame in range(n_frames):
            if idx_frame == frame_range[1]:
                break
            if (frame_range[0] <= idx_frame) and ((idx_frame - frame_range[0])%stride == 0): 
                # skip header
                f.readline()
                # save data into string
                data_str = ""
                for i in range(config.n_beads):
                    data_str += f.readline()        
                # read in data
                frame = np.genfromtxt(StringIO(data_str), delimiter=" ")
                forces[idx_traj] = np.array(frame.tolist())
                idx_traj += 1
            else:
                # skip frame
                for i in range(config.n_beads + 1):
                    f.readline()
    f.close()
            
    return forces

def load_trajectory(config: Optional[Config],
                    frame_range: list = None,
                    frame: int = None,
                    stride: int = 1):
    """
    Loads Trajectory into numpy array forthe given range of frames.
    If frame_range is None, the whole trajectory is loaded.
    """
    
    # if config is given as a path, load it
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    # get traj path
    fname_traj = get_path(config, filetype="trajectory")
    
    # get number of frames
    n_frames = get_number_of_frames(config)
    
    if frame is not None:
        frame_range = list([frame, frame+1])
    
    # handle frame range input
    if frame_range is None:
        frame_range = list([0, n_frames])
    if frame_range[0] == None:
        frame_range[0] = 0
    if frame_range[1] == None:
        frame_range[1] = n_frames
    if frame_range[1] > n_frames:
        frame_range[1] = n_frames
        print(f"Warning: Frame range exceeds trajectory length. Setting frame_range[1] to {n_frames}")
    if frame_range[0] < -n_frames:
        raise ValueError(f"Frame range is not valid: index 0 exceeds trajectory length of {n_frames} frames")
    if frame_range[0] < 0:
        frame_range[0] = n_frames + frame_range[0]
    if frame_range[1] <= -n_frames:
        raise ValueError(f"Frame range is not valid: index 1 exceeds trajectory length of {n_frames} frames")
    if frame_range[1] < 0:
        frame_range[1] = n_frames + frame_range[1]
    if frame_range[0] >= frame_range[1]:
        raise ValueError("Frame range is not valid: index 0 is larger than index 1")
    
    #define a np.dtype for gro array/dataset (hard-coded for now)
    gro_dt = np.dtype([('col1', 'S4'), ('col2', 'S4'), ('col3', int), 
                    ('col4', float), ('col5', float), ('col6', float)])

    # initialize trajectory array
    traj = np.zeros(((frame_range[1] - frame_range[0] - 1)//stride + 1, config.n_beads, 3))
    
    idx_traj = 0
    with open(fname_traj, "r") as f:
        
        for idx_frame in range(n_frames):
            if idx_frame == frame_range[1]:
                break
            if (frame_range[0] <= idx_frame) and ((idx_frame - frame_range[0])%stride == 0): 
                # skip header
                for i in range(2):
                    f.readline()
                # save data into string
                data_str = ""
                for i in range(config.n_beads):
                    data_str += f.readline()        
                # skip box size
                f.readline()
                
                # check if data string is empty -> end of file
                if data_str == "":
                    print(f"Warning: Trajectory of {config.name_sys} unexpectedly ended at frame {idx_frame}.")
                    traj = traj[:idx_traj]
                    break
                
                # read in data
                frame = np.genfromtxt(StringIO(data_str), dtype=gro_dt, usecols=(3, 4, 5))
                traj[idx_traj] = np.array(frame.tolist())
                idx_traj += 1
            else:
                # skip frame
                for i in range(config.n_beads + 3):
                    f.readline()
    
        # check if there are more frames in the file
        if frame_range[1] == n_frames:
            # skip frames that are not read beacause of stride
            for i in range((frame_range[1]-frame_range[0])%stride):
                for j in range(config.n_beads + 3):
                    f.readline() # skip line
            data_str = ""
            for i in range(config.n_beads + 3):
                data_str += f.readline()
            if data_str != "":
                print(f"Warning: Trajectory of {config.name_sys} unexpectedly continued after frame {frame_range[1]}.")
                print(data_str)
    
    f.close()
            
    return traj

def save_trajectory(config: Optional[Config],
                    trajectory: np.ndarray,
                    fname: str = None,
                    type: str = "gro",
                    overwrite: bool = False):
    if isinstance(config, str):
        config = Config.from_toml(config)
    if len(trajectory.shape) != 3:
        raise ValueError("trajectory must be a 3d array")
    if fname is None:
        fname = get_path(config, filetype="trajectory", overwrite=overwrite)
    if type == "gro":
        n_atoms = len(trajectory[0])
        for i, frame in enumerate(trajectory):
            _write_frame_gro(n_atoms, frame, i, fname)
            
def cut_trajectory(config: Optional[Config],
                   frame_range: list = None,
                   bead_indices: list = None,
                   fname: str = None,
                   type: str = "gro"):
    """
    saves traejctory as a new gro file with the given frame range.
    """
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    fname_traj = get_path(config, filetype="trajectory", overwrite=True)
    
    # get number of frames
    n_frames = get_number_of_frames(config)
    
    if frame_range is None:
        frame_range = (0, n_frames)
    if frame_range[0] == None:
        frame_range[0] = 0
    if frame_range[1] == None:
        frame_range[1] = n_frames
    if frame_range[0] < 0 or frame_range[1] > n_frames:
        raise ValueError("frame range out of bounds")

    traj = load_trajectory(config, frame_range=frame_range)
    
    if bead_indices is not None:
        traj = traj[:, bead_indices, :]
        
    if fname is None:
        fname = Path(fname_traj).parent / f"traj_{config.name_sys}_frame_{frame_range[0]}_to_{frame_range[1]}_nbeads_{len(bead_indices)}.gro"
    
    if type == "gro":
        save_trajectory(config, traj, fname=fname)

    
def _write_frame_gro(n_atoms, coordinates, time, fname, comment="trajectory", box=None, precision=3):
    f = open(fname, "a")
    comment += ', t= %s' % time
    varwidth = precision + 5
    fmt = '%%5d%%-5s%%5s%%5d%%%d.%df%%%d.%df%%%d.%df' % (
            varwidth, precision, varwidth, precision, varwidth, precision)
    lines = [comment, ' %d' % n_atoms]
    for i in range(n_atoms):
        lines.append(fmt % (i+1, "HET", "CA", i+1,
                            coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]))
    lines.append('%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f' % (0,0,0,0,0,0,0,0,0))
    f.write('\n'.join(lines))
    f.write('\n')
    f.close()

def get_number_of_frames(config: Optional[Config]):
    return int(config.steps/config.stride)

    
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

    # TODO DOES THIS MAKE SENSE ???????????
    # apply minimum image convetion
    directions -= config.lbox*np.round(directions/config.lbox)

    # calculate distances and apply interaction cutoff
    distances = np.linalg.norm(directions, axis=2)

    # apply cutoff
    mask = distances < config.cutoff_pbc
    
    # only return distances, that lie within the interaction cutoff
    return distances[mask]     

def delete_system(config: Optional[Config],
                  only_results: bool = False):
    """
    Deletes every file of a scpecified system. Use with care.
    """
    
    dir_dict = {"trajectory":           ("/trajectories/traj_",                 ".gro"),
                "config":               ("/configs/cfg_",                       ".toml"),
                "parameters":           ("/parameters/param_",                  ".toml"),
                "init_pos":             ("/initial_positions/xyz_",             ".npy"),
                "rdf":                  ("/results/rdf/rdf_",                   ".npy"),
                "structure_factor":     ("/results/structure_factor/Sq_",       ".npy"),
                "structure_factor_rdf": ("/results/structure_factor/Sq_rdf_",       ".npy"),
                "stress_tensor":        ("/results/stress_tensor/sigma_",       ".npy"),
                "forces":               ("/forces/forces_",                     ".txt")}
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    if not only_results:    
        proceed = "y" == input(f"Are you sure you want to delete system {config.name_sys} and all its related files? (y/n) ")
        if proceed:
            for key in dir_dict.keys():
                fname = get_path(config, filetype=key)
                if os.path.exists(fname):
                    os.remove(fname)
            print(f"System {config.name_sys} deleted.")
    else:
        proceed = "y" == input(f"Are you sure you want to delete all results of system {config.name_sys}? (y/n) ")
        if proceed:
            for key in ["rdf", "structure_factor", "structure_factor_rdf", "stress_tensor"]:
                fname = get_path(config, filetype=key)
                if os.path.exists(fname):
                    os.remove(fname)
            print(f"Results of system {config.name_sys} deleted.")
def save_config():
    # TODO
    print("function does nothing (todo)")
    return

    
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