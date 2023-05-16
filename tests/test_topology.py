from config import Config
from topology import Topology

cfg_path = "/home/janmak98/mucus/systems/test_var_params/configs/cfg_test_mucus.toml"
cfg = Config.from_toml(cfg_path)
top = Topology(cfg)

# check if every tag index could be reached
n = cfg.number_of_beads
for i in range(n):
    for j in range(n):
        n, m = top.get_tag_indices(i, j)


