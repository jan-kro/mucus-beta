import sys
import time
import datetime

from .config import Config
from .system import System


def run_sim(type: str = "box", fname_config: str = None):
    
    if fname_config is None:
        config = Config.from_toml(sys.argv[1])
    else:
        config = Config.from_toml(fname_config)
        
    p = System(config)
    
    if type == "box":
        p.create_box()
    if type == "chain":
        p.create_chain()
    
    print("starting a simulation with following parameters:")
    p.print_sim_info()
    
    now = time.localtime()
    now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"
    print("\nsimulation started ", now_str)
    
    p.simulate()

    now = time.localtime()
    now_str = f"{now.tm_mon:d}.{now.tm_mday}.{now.tm_year}  {now.tm_hour}:{now.tm_min}:{now.tm_sec}"

    print("simulation finished ", now_str)

    print("simulation time: ", datetime.timedelta(seconds=round(p.config.simulation_time)))
    
    print("\nsave system...")
    p.save_system()
    print("\ndone")