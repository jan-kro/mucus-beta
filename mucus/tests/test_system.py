from system import System
from config import Config

cfg_path = "/home/janmak98/mucus/mucus/tests/test_files/connected-S-mesh/configs/cfg_connected-S-mesh.toml"
cfg = Config.from_toml(cfg_path)
sys = System(cfg)
print("simulating...")
sys.print_sim_info()
sys.simulate()
print("done")