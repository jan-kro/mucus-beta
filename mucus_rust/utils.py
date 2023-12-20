import enum
import os
import toml
import numpy as np
import rust_mucus as rmc
import h5py
from .config import Config
from .topology import Topology
from typing import Optional
from pathlib import Path
from io import StringIO
from tqdm import tqdm


# TODO USE THIS INSTEAD OF STRINGS TO SPECIFY DATASET TYPE
class Filetypes(enum.Enum):
    Trajectory = "trajectory"
    Forces = "forces"
    Distances = "distances"
    Rdf = "rdf"
    StructureFactor = "structure_factor"
    StructureFactorRdf = "structure_factor_rdf"
    StressTensor = "stress_tensor"
    Msd = "msd"
    Results = "results"
    Config = "config"
    Parameters = "parameters"
    InitPos = "init_pos"


class ResultsFiletypes(enum.Enum):
    Trajectory = Filetypes.Trajectory.value
    Forces = Filetypes.Forces.value
    Distances = Filetypes.Distances.value
    Rdf = Filetypes.Rdf.value
    StructureFactor = Filetypes.StructureFactor.value
    StructureFactorRdf = Filetypes.StructureFactorRdf.value
    StressTensor = Filetypes.StressTensor.value
    Msd = Filetypes.Msd.value

class ParametersFiletypes(enum.Enum):
    Config = Filetypes.Config.value
    Parameters = Filetypes.Parameters.value
    InitPos = Filetypes.InitPos.value
    ParticleRadius = "r_particles"
    ParticleCharge = "q_particles"
    Mobility = "mobilities"
    ForceConstant = "force_constants"
    LjEpsilon = "epsilon_LJ"
    LjSigma = "sigma_LJ"
    BondTable = "bond_table"
    BondDistance = "r0_bonds"
    Tags = "tags"


class ConfigParameters(enum.Enum):
    Steps = "steps"
    Stride = "stride"
    NumParticles = "n_particles"
    Timestep = "timestep"
    LengthScaleIn_nm = "r0_nm"
    LjCutoff = "cutoff_LJ"
    BjerrumLength = "lB_debye"
    SaltConcentration = "c_S"
    DebyeCutoff = "cutoff_debye"
    BoxLength = "lbox"
    Pbc = "pbc"
    PbcCutoff = "cutoff_pbc"
    WriteTraj = "write_traj"
    Cunksize = "chunksize"
    WriteForces = "write_forces"
    WriteDistances = "write_distances"
    Cwd = "cwd"
    NameSys = "name_sys"
    DirSys = "dir_sys"
    SimulationTime = "simulation_time"

# Enum ForceType 
#     None
#     Harmonic{k: f64},
#     DoubleWell{k: f64, a: f64},
#     LinearInterpolation{values: Array<f64>, fe: Array<f64>},
    

# func get_force(force_type: ForceType, r: f64) -> f64 {
#     match force_type {
#         ForceType::None => {
#             0.0
#         },
#         ForceType::Harmonic{k_harm} => {
#             -k_harm * r
#         },
#         ForceType::DoubleWell{k, a} => {
#             -k*(r-a)*(r+a)
#         },
#         ForceType::LinearInterpolation{values, fe} => {
#             -k*(r-a)*(r+a)
#         },
#     }
# }
    
# dset_to_write = Dataset.Trajectory
# print(file[dset_to_write.value])


def _dir_dict():
    
    dir_dict = {"trajectory":           ("/trajectories/traj_",                 ".gro"),
                "config":               ("/configs/cfg_",                       ".toml"),
                "parameters":           ("/parameters/param_",                  ".toml"),
                "init_pos":             ("/initial_positions/xyz_",             ".npy"),
                "rdf":                  ("/results/rdf/rdf_",                   ".npy"),
                "structure_factor":     ("/results/structure_factor/Sq_",       ".npy"),
                "structure_factor_rdf": ("/results/structure_factor/Sq_rdf_",   ".npy"),
                "stress_tensor":        ("/results/stress_tensor/sigma_",       ".npy"),
                "msd":                  ("/results/msd/msd_",                   ".npy"),
                "forces":               ("/forces/forces_",                     ".txt"),
                "results":              ("/results/results_",                   ".hdf5")}

    return dir_dict

def _get_results_filetypes():
    return ["trajectory", "forces", "distances", "rdf", "structure_factor", "structure_factor_rdf", "stress_tensor", "msd"]

# TODO THIS DOES NOTHING REMOVE
def _create_results(fname: str = None, 
                    dataset: str = "trajectory",
                    shape: list = None,
                    maxshape: list = None,
                    dtype: str = "float64"):
    """
    Creates the h5 file containing a dataset of the specified filetype. If the dataset does not exist, it is created.
    If the h5 file does not exist, it is created. 
    
    datasets:
        "trajectory"
        "forces"
        "distances"
        "rdf"
        "structure_factor"
        "structure_factor_rdf"
        "stress_tensor"
    
    Parameters:
    ----------
    Config: Config object
        Containing all information of the system to analyze
    dataset: str
        Type of dataset to load
    shape: list
        Shape of the dataset (only used if dataset is not yet contained in the h5 file)
    maxshape: list
        Maxshape of the dataset (only used if dataset is not yet contained in the h5 file)
    dtype: str
        Dtype of the dataset (only used if dataset is not yet contained in the h5 file)
    overwrite: bool
        If False, the system name is appended with a version number if the h5 file already exists
        and a new h5 file is created.
    """
    
    result_filetypes = _get_results_filetypes()
        
    h5 = h5py.File(fname, "a")
    
    # check id dataset type is valid, so that there is not wrong dataset created by accident
    if not dataset in result_filetypes:
        raise ValueError(f"{dataset} is not a valid dataset type")
    
    # create dataset if it is not yet contained in the h5 file
    if dataset not in h5.keys():
        ds = h5.create_dataset(dataset, shape=shape, maxshape=maxshape, dtype=dtype)
        
    h5.close()
    
def get_path(Config: Optional[Config], 
             filetype: str = "trajectory",
             overwrite: bool = True,
             ): # **kwargs):
    """
    Creates an outfile str for a specified filetype that includes the absolute directory and filename. 
    If the directory does not exist, it is created. The overwrite call checks if the system name already 
    exists in the output directory. If it does, a version str ("_vXX") is appended to the system name. 
    
    filetype:
        "trajectory"
        "config"
        "init_pos"
        "parameters"
        "rdf"
        "structure_factor"
        "forces"
        "results"
    """
    

    dir_dict = _dir_dict()
        
    if filetype not in dir_dict.keys():
        raise ValueError(f"filetype {filetype} not valid")
    
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
            # if file exist update version until no file with that version exists
            while os.path.exists(base + "/" + name + split_arg + str(version) + ext):
                version += 1
            # update fname with new version
            fname = base + "/" + name + split_arg + str(version) + ext

    # if parent path doesnt exist, create it
    if not os.path.exists(Path(fname).parent):
        os.makedirs(Path(fname).parent)
    
    # if filetype == "results":
    #     _create_results(fname, **kwargs)
            
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

    # get number of frames
    n_frames = get_number_of_frames(config)
    frame_range = _validate_frame_range(frame_range, n_frames)
    
    # get traj path
    h5path = get_path(config, filetype="results")
    
    if os.path.exists(h5path):
        with h5py.File(h5path, "r") as h5:
            if "forces" in h5.keys():
                return h5["forces"][frame_range[0]:frame_range[1]:stride]

    # if no h5 file exists, load forces from txt file
    fname_traj = get_path(config, filetype="forces")
    
    # TODO create force file using rust function
    # if not os.path.exists(fname_traj):
    #     print(f"forces file {fname_traj} not found")
    #     print(f"Creating forces file from {config.name_sys}.gro ...")
        
    #     # create sys and load trajectory
    #     traj = load_trajectory(config)
    #     sys = System(config)
        
    #     f_force = open(fname_traj, "a")
    #     forces_frame = np.zeros((config.n_particles, 3))
    #     for i, frame in tqdm(enumerate(traj)):
    #         sys.set_positions(frame)
    #         forces_frame.fill(0)
    #         rmc.get_forces(frame, 
    #                        sys.tags, 
    #                        self.bond_table, 
    #                        self.topology.force_constant_nn, 
    #                        self.topology.r0_bond, 
    #                        self.topology.sigma_lj,
    #                        self.topology.epsilon_lj,      
    #                        self.charges,          
    #                        self.config.lB_debye,         
    #                        self.B_debye,          
    #                        self.forces, 
    #                        self.box_length, 
    #                        self.config.cutoff_pbc**2, 
    #                        self.n_particles, 
    #                        3, 
    #                        True, 
    #                        True, 
    #                        False)
    #         sys.write_frame_force(sys.forces, f_force, i)

    #     f_force.close()
    #     print("Done.")
    
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
    forces = np.zeros(((frame_range[1] - frame_range[0] - 1)//stride + 1, config.n_particles, 3))
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
                for i in range(config.n_particles):
                    data_str += f.readline()        
                # read in data
                frame = np.genfromtxt(StringIO(data_str), delimiter=" ")
                forces[idx_traj] = np.array(frame.tolist())
                idx_traj += 1
            else:
                # skip frame
                for i in range(config.n_particles + 1):
                    f.readline()
    f.close()
            
    return forces

def load_results(config: Optional[Config],
                 filetype: str = ResultsFiletypes.Trajectory.value,
                 frame_range: list = None,
                 stride: int = 1):
    
    if filetype not in [i.value for i in ResultsFiletypes]:
        raise ValueError(f"Filetype '{filetype}' is not a valid filetype.\n Valid filetypes are {[i.value for i in ResultsFiletypes]}")

    # if config is given as a path, load it
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    # get number of frames
    n_frames = get_number_of_frames(config)
    
    # handle frame range input
    frame_range = _validate_frame_range(frame_range, n_frames)
    
    # get h5py filename
    fname_h5 = get_path(config, filetype=Filetypes.Results.value)
    
    if not os.path.exists(fname_h5):
        raise FileNotFoundError(f"Results file '{fname_h5}' not found")
    
    with h5py.File(fname_h5, "r") as h5_file:
        if filetype in h5_file.keys():
            try:
                return h5_file[filetype][frame_range[0]:frame_range[1]:stride]
            except:
                raise ValueError(f"Frame range {frame_range} is not valid.\nThe shape dataset '{filetype}' is {h5_file[filetype].shape}.")
        else:
            raise ValueError(f"Dataset '{filetype}' is not contained in.\nValid filetypes are {[i.value for i in ResultsFiletypes]}")
    

# TODO THIS IS INSANE TO LOAD THE WHOLE TRAJ INTO MEMORY WHY WOULD YOU DO THAT??
def load_trajectory_h5(config: Optional[Config],
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
    
    if frame is not None:
        frame_range = list([frame, frame+1])
    
    n_frames = get_number_of_frames(config)
    frame_range = _validate_frame_range(frame_range, n_frames)
    
    # get traj path
    fname_traj = get_path(config, filetype="results")
    
    if not os.path.exists(fname_traj):
        print(f"results file {fname_traj} not found")
        load_gro = "y" == input(f"Load trajectory from {get_path(config, filetype='trajectory')}? (y/n) ")
        if load_gro:
            trajectory = load_trajectory(config, frame_range=frame_range, stride=stride)
            save_h5 = "y" == input(f"Save trajectory to {fname_traj}? (y/n) ")
            if save_h5:
                with h5py.File(fname_traj, "a") as h5:
                    h5.create_dataset("trajectory", data=trajectory)
            return trajectory
        else:
            raise FileNotFoundError(f"trajectory file {fname_traj} not found")
    traj_h5 = h5py.File(fname_traj, "r")
    trajectory = traj_h5["trajectory"][frame_range[0]:frame_range[1]:stride]
    
    return trajectory
    
def _validate_frame_range(frame_range, n_frames):
    if frame_range is None:
        frame_range = [0, n_frames]
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
    
    return frame_range
        
    
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
    frame_range = _validate_frame_range(frame_range, n_frames)
    
    #define a np.dtype for gro array/dataset (hard-coded for now)
    gro_dt = np.dtype([('col1', 'S4'), ('col2', 'S4'), ('col3', int), 
                    ('col4', float), ('col5', float), ('col6', float)])

    # initialize trajectory array
    traj = np.zeros(((frame_range[1] - frame_range[0] - 1)//stride + 1, config.n_particles, 3))
    
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
                for i in range(config.n_particles):
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
                for i in range(config.n_particles + 3):
                    f.readline()
    
        # check if there are more frames in the file
        if frame_range[1] == n_frames:
            # skip frames that are not read beacause of stride
            for i in range((frame_range[1]-frame_range[0])%stride):
                for j in range(config.n_particles + 3):
                    f.readline() # skip line
            data_str = ""
            for i in range(config.n_particles + 3):
                data_str += f.readline()
            if data_str != "":
                print(f"Warning: Trajectory of {config.name_sys} unexpectedly continued after frame {frame_range[1]}.")
                print(data_str)
            
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


def traj_h5_to_gro(config: Optional[Config],
                   frame_range: list = None,
                   stride: int = 1,
                   fname: str = None,
                   overwrite: bool = False):
    """
    Saves trajectory from h5 file as a gro file.
    """
    
    if isinstance(config, str):
         config = Config.from_toml(config)
    
    # get number of frames
    n_frames = get_number_of_frames(config)
    
    # get traj path
    if fname is None:
        fname_traj = get_path(config, filetype="trajectory")
    
        if frame_range != None:
            frame_range = _validate_frame_range(frame_range, n_frames)
            fname_traj = fname_traj.split(".gro")[0] + f"_frame_{frame_range[0]}_to_{frame_range[1]}_stride{stride}.gro"       
    else:
        fname_traj = fname
            
    # handle frame range input
    frame_range = _validate_frame_range(frame_range, n_frames)
    
    # get h5py filename
    fname_h5 = get_path(config, filetype="results")
    
    if not os.path.exists(fname_h5):
         raise FileNotFoundError(f"Results file '{fname_h5}' not found")
    
    with h5py.File(fname_h5, "r") as h5_file:
         if "trajectory" in h5_file.keys():
               trajectory = h5_file["trajectory"][frame_range[0]:frame_range[1]:stride]
               save_trajectory(config, trajectory, fname=fname_traj, overwrite=overwrite, type="gro")
         else:
               raise ValueError(f"Dataset 'trajectory' is not contained in {fname_h5}.")
            
def cut_trajectory(config: Optional[Config],
                   frame_range: list = None,
                   particle_indices: list = None,
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
    
    if particle_indices is not None:
        traj = traj[:, particle_indices, :]
        
    if fname is None:
        fname = Path(fname_traj).parent / f"traj_{config.name_sys}_frame_{frame_range[0]}_to_{frame_range[1]}_nparticles_{len(particle_indices)}.gro"
    
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
    r_left = np.tile(frame, (config.n_particles, 1, 1)) # repeats vector along third dimension len(a) times
    r_right = np.reshape(np.repeat(frame, config.n_particles, 0), (config.n_particles, config.n_particles, 3)) # does the same but "flipped"

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
                  only_results: bool = False,
                  exceptions: list = []):
    """
    Deletes every file of a scpecified system. Use with care.
    """
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    if only_results:
        exceptions = ["config", "init_pos", "parameters"] + list(exceptions)
        print(exceptions)

 
    proceed = "y" == input(f"Are you sure you want to delete system {config.name_sys} and all its related files (exceptions: {exceptions})? (y/n) ")
    if proceed:
        dir_dict = _dir_dict()
        for key in dir_dict.keys():
            if key not in exceptions:
                fname = get_path(config, filetype=key)
                if os.path.exists(fname):
                    os.remove(fname)
                    print(f"File {fname} deleted.")
        print(f"System {config.name_sys} deleted.\n")
    else:
        print("Aborted.\n")

def update_system_version(config: Optional[Config],
                          Topology: Optional[Topology],
                          only_results: bool = False):
    """
    Saves config, initial positions and topology of a system,
    using the specified version in the sys_name specified in the config.
    """
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    print("function does nothing (todo)")
    return

def get_timestep_seconds(config: Optional[Config],
                        monomer_tag = 0):
        """
        returns the timestep of the current trajectory in seconds (stride and cfg-timestep included)
        """
        if isinstance(config, str):
            config = Config.from_toml(config)
        top = Topology(config)
        
        mu = top.mobility[monomer_tag]  # mobility of the system in reduced units
        a = 1e-9*config.r0_nm           # m, reduced legth scale: PEG monomere radius
        r = 1*a                         # m, particle radius

        eta_w = 8.53e-4 # Pa*s
        kB = 1.380649e-23 # m^2 kg s^-2 K^-1
        T = 300 # K
        mu_0 = kB*T/(6*np.pi*eta_w*r*a**2) 

        dt_step = mu/mu_0
        # multiply stepwise timestep with the simulation stride 
        dt = config.stride*dt_step*config.timestep # s
        
        return dt
    
def correlation(a,b=None,subtract_mean=False):
    """
    henriks correlation function
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