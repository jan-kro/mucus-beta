from mucus.system import System
from mucus.config import Config

cfg_path = "tests/connected-S-mesh/configs/cfg_connected-S-mesh.toml"
cfg = Config.from_toml(cfg_path)
sys = System(cfg)
print("simulating...")
sys.print_sim_info()
sys.simulate()
print("done")