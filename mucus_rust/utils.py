import enum
import os
import toml
import numpy as np
import rust_mucus as rmc
import h5py
from .config import Config
from typing import Optional
from pathlib import Path
from io import StringIO
from tqdm import tqdm

# TODO USE THIS INSTEAD OF STRINGS TO SPECIFY DATASET TYPE

class Filetypes(enum.Enum):
    Trajectory          = "trajectory"
    TrajectoryGro       = "trajectory_gro"
    Forces              = "forces"
    Distances           = "distances"
    Rdf                 = "rdf"
    StructureFactor     = "structure_factor"
    StructureFactorRdf  = "structure_factor_rdf"
    StressTensor        = "stress_tensor"
    Msd                 = "msd"
    Results             = "results"
    Config              = "config"
    Parameters          = "parameters"
    InitPos             = "init_pos"


class ResultsFiletypes(enum.Enum):
    Trajectory          = Filetypes.Trajectory.value
    TrajectoryGro       = Filetypes.TrajectoryGro.value
    Forces              = Filetypes.Forces.value
    Distances           = Filetypes.Distances.value
    Rdf                 = Filetypes.Rdf.value
    StructureFactor     = Filetypes.StructureFactor.value
    StructureFactorRdf  = Filetypes.StructureFactorRdf.value
    StressTensor        = Filetypes.StressTensor.value
    Msd                 = Filetypes.Msd.value

class ParametersFiletypes(enum.Enum):
    Config              = Filetypes.Config.value
    Parameters          = Filetypes.Parameters.value
    InitPos             = Filetypes.InitPos.value
    ParticleRadius      = "r_particles"
    ParticleCharge      = "q_particles"
    Mobility            = "mobilities"
    ForceConstant       = "force_constants"
    LjEpsilon           = "epsilon_LJ"
    LjSigma             = "sigma_LJ"
    BondTable           = "bond_table"
    BondDistance        = "r0_bonds"
    Tags                = "tags"


class ConfigParameters(enum.Enum):
    Steps               = "steps"
    Stride              = "stride"
    NumParticles        = "n_particles"
    Timestep            = "timestep"
    LengthScaleIn_nm    = "r0_nm"
    LjCutoff            = "cutoff_LJ"
    BjerrumLength       = "lB_debye"
    SaltConcentration   = "c_S"
    DebyeCutoff         = "cutoff_debye"
    BoxLength           = "lbox"
    Pbc                 = "pbc"
    PbcCutoff           = "cutoff_pbc"
    WriteTraj           = "write_traj"
    Cunksize            = "chunksize"
    WriteForces         = "write_forces"
    WriteDistances      = "write_distances"
    Cwd                 = "cwd"
    NameSys             = "name_sys"
    DirSys              = "dir_sys"
    SimulationTime      = "simulation_time"

class NatrualConstantsSI(enum.Enum):
    kB = 1.380649e-23 # m^2 kg s^-2 K^-1
    T = 300 # K
    
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

# TODO somehow make this more elegant (also Enum class?)
def _dir_dict():
    
    dir_dict = {"trajectory_gro":       ("/snapshots/traj_",                    ".gro"),
                "config":               ("/configs/cfg_",                       ".toml"),
                "parameters":           ("/parameters/param_",                  ".toml"),
                "init_pos":             ("/initial_positions/xyz_",             ".npy"),
                "rdf":                  ("/results/rdf_",                       ".hdf5"),
                "structure_factor":     ("/results/Sq_",                        ".hdf5"),
                "structure_factor_rdf": ("/results/Sq_rdf_",                    ".hdf5"),
                "stress_tensor":        ("/results/sigma_",                     ".hdf5"),
                "msd":                  ("/results/msd_",                       ".hdf5"),
                "trajectory":           ("/results/traj_",                      ".hdf5"),
                "forces":               ("/results/forces_",                    ".hdf5"),
                "distances":            ("/results/distances_",                 ".hdf5"),
                "results":              ("/results/results_",                   ".hdf5")}

    return dir_dict

    
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

def convert_trajectory(config: Optional[Config],
                    trajectory: np.ndarray,
                    fname: str = None,
                    trajtype: str = "gro",
                    overwrite: bool = False):
    """
    Converts trajectory (3d nd.array) to a specified fieltype
    (For now only type "gro" is possible)
    """
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    if len(trajectory.shape) != 3:
        raise ValueError("trajectory must be a 3d array")
    if trajtype == "gro":
        if fname is None:
            fname = get_path(config, filetype=ResultsFiletypes.TrajectoryGro.value, overwrite=overwrite)
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
        fname_traj = get_path(config, filetype=ResultsFiletypes.TrajectoryGro.value)
    
        if frame_range != None:
            frame_range = _validate_frame_range(frame_range, n_frames)
            fname_traj = fname_traj.split(".gro")[0] + f"_frame_{frame_range[0]}_to_{frame_range[1]}_stride{stride}.gro"       
    else:
        fname_traj = fname
            
    # handle frame range input
    frame_range = _validate_frame_range(frame_range, n_frames)
    
    # get h5py filename
    fname_h5 = get_path(config, filetype=ResultsFiletypes.Trajectory.value)
    
    if not os.path.exists(fname_h5):
         raise FileNotFoundError(f"Trajectory file '{fname_h5}' not found")
    
    with h5py.File(fname_h5, "r") as h5_file:
        trajectory = h5_file[ResultsFiletypes.Trajectory.value][frame_range[0]:frame_range[1]:stride]
        convert_trajectory(config, trajectory, fname=fname_traj, overwrite=overwrite, trajtype="gro")


    
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

def get_timestep_seconds(config: Optional[Config],
                        monomer_tag = 0):
        """
        returns the timestep of the current trajectory in seconds (stride and cfg-timestep included)
        """
        if isinstance(config, str):
            config = Config.from_toml(config)
            
        params = toml.load(open(get_path(config, "parameters"), encoding="UTF-8"))
        mobility = np.array(params["mobilities"])
        
        mu = mobility[monomer_tag]  # mobility of the system in reduced units
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

# TODO THIS IS INSANE TO LOAD THE WHOLE TRAJ INTO MEMORY WHY WOULD YOU DO THAT??
# def load_trajectory_h5(config: Optional[Config],
#                        frame_range: list = None,
#                        frame: int = None,
#                        stride: int = 1,
#                        locking=None):
    
#     """
#     Loads Trajectory into numpy array forthe given range of frames.
#     If frame_range is None, the whole trajectory is loaded.
#     """
    
#     # if config is given as a path, load it
#     if isinstance(config, str):
#         config = Config.from_toml(config)
    
#     if frame is not None:
#         frame_range = list([frame, frame+1])
    
#     n_frames = get_number_of_frames(config)
#     frame_range = _validate_frame_range(frame_range, n_frames)
    
#     # get traj path
#     fname_traj = get_path(config, filetype="trajectory")
    
#     if not os.path.exists(fname_traj):
#         print(f"results file {fname_traj} not found")
#         load_gro = "y" == input(f"Load trajectory from {get_path(config, filetype='trajectory_gro')}? (y/n) ")
#         if load_gro:
#             trajectory = load_trajectory(config, frame_range=frame_range, stride=stride)
#             save_h5 = "y" == input(f"Save trajectory to {fname_traj}? (y/n) ")
#             if save_h5:
#                 with h5py.File(fname_traj, "a", locking=locking) as h5:
#                     h5.create_dataset("trajectory", data=trajectory)
#             return trajectory
#         else:
#             raise FileNotFoundError(f"trajectory file {fname_traj} not found")
#     traj_h5 = h5py.File(fname_traj, "r",locking=locking)
#     trajectory = traj_h5["trajectory"][frame_range[0]:frame_range[1]:stride]
    
#     return trajectory


# def update_system_version(config: Optional[Config],
#                           Topology: Optional[Topology],
#                           only_results: bool = False):
#     """
#     Saves config, initial positions and topology of a system,
#     using the specified version in the sys_name specified in the config.
#     """
    
#     if isinstance(config, str):
#         config = Config.from_toml(config)
    
#     print("function does nothing (todo)")
#     return