from mucus_rust.system import System
from mucus_rust.config import Config
from mucus_rust.utils import delete_system


cfg_path = "test"
cfg = Config.from_toml(cfg_path)

# delete all results
delete_system(cfg, only_results=True)

sys = System(cfg)
print("\nsimulating...\n")
sys.print_sim_info()
sys.simulate()
print("done")

