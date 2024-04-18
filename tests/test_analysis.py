import mucus_rust as mcr
import matplotlib.pyplot as plt
import numpy as np
import h5py

cfg_path = "test"
cfg = mcr.Config.from_toml(cfg_path)

fname = mcr.utils.get_path(cfg, "results", overwrite=True)
h5_file = h5py.File(fname, "a")
if "structure_factor" in h5_file.keys():
    del h5_file["structure_factor"]
if "structure_factor_params" in h5_file.keys():
    del h5_file["structure_factor_params"]
h5_file.close()

print("Initializing analysis object...")
sys = mcr.Analysis(cfg, frame_range=[0, 100])
print("done\n")

print("Test get timestep seconds...")
dt = sys.get_timestep_seconds(monomer_tag=0)

print("Calculate Structure Factor... ")
q, Sq = sys.structure_factor(return_all=True)
print("done\n")

print("Calculate Structure Factor again to see if its calculation is already recognized... ")
q, Sq = sys.structure_factor(return_all=True)
print("done\n")

print("Create second analysis object with different stride and calculate structure factor again... ")
sys2 = mcr.Analysis(cfg, stride=2)
q2, Sq2 = sys2.structure_factor(return_all=True)
print("done\n") 

fig, ax = plt.subplots()
ax.plot(q, Sq, label="stride=1")
ax.plot(q2, Sq2, label="stride=2")
ax.legend()
plt.show()

print("Test stress tensor calculation...")
sigma_t = sys.virial_stress_tensor(save = True, return_all = True)
print("done\n")

fname = mcr.utils.get_path(cfg, "results", overwrite=True)
h5_file = h5py.File(fname, "a")
print("Final h5 file keys: ", [key for key in h5_file.keys()])
h5_file.close()

