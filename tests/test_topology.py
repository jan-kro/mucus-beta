from mucus_rust.config import Config
from mucus_rust.topology import Topology
import toml
import numpy as np


cfg_path = "test"
cfg = Config.from_toml(cfg_path)

print("Initializing topology...")
top = Topology(cfg)
top2 = Topology(cfg)

print("done\n")
print("Test setting new parameters...")

params_toml = toml.load(open(top._get_path(cfg, "parameters"), encoding="UTF-8"))

for key, value in params_toml.items():
    top.set_parameter(np.array(value), key=key)

same_params = True

for key1, value1, key2, value2 in zip(top.__dict__.keys(), top.__dict__.values(), top2.__dict__.keys(), top2.__dict__.values()):
    if not np.all(value1 == value2):
        same_params = False
        print(f"{key1} and {key2} are not the same")

print(f"All parameters are the same: {same_params}")
print("done\n")

print("check if every tag index could be reached...")
n = cfg.n_particles
for i in range(n):
    for j in range(n):
        nt, mt = top.get_tags(i, j)
print("done\n")

print("Test saving the topology...")
top.save(cfg)
print("done\n")

print("Test loading the topology again...")
top3 = Topology(cfg)

for key1, value1, key2, value2 in zip(top.__dict__.keys(), top.__dict__.values(), top3.__dict__.keys(), top3.__dict__.values()):
    if not np.all(value1 == value2):
        same_params = False
        print(f"{key1} and {key2} are not the same")

print(f"All parameters are the same: {same_params}")
print("done\n")


