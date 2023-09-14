import h5py
import mucus_rust as mc
from time import time
import numpy as np

cfg_path = "/net/storage/janmak98/masterthesis/output/test_systems/configs/cfg_mesh_tracer_6a_uncharged.toml"
cfg = mc.Config.from_toml(cfg_path)
top = mc.Topology(cfg)

tt = time()
traj = mc.utils.load_trajectory(cfg)
tt = time() - tt

ttso = time()
traj_sliced = mc.utils.load_trajectory(cfg, frame_range=[100,200])
ttso = time() - ttso

# print("Shape of test trajectory:")
# print(traj.shape, "\n")

fname_traj = "/home/janmak98/mucus_rust_test/tests/testtraj.hdf5"

n_frames = mc.utils.get_number_of_frames(cfg)
traj_shape = (n_frames, cfg.n_beads, 3)
# print("Shape of trajectory (calculated):")
# print(traj_shape, "\n")
traj_h5 = h5py.File(fname_traj, "w")
traj_h5.create_dataset("trajectory", traj_shape, dtype="float64")

# print("frame number 123 of created dataset:")
# print(traj_h5["trajectory"][123], "\n")

stride = 93

idx_frame = 1*stride
while idx_frame < n_frames:
    traj_h5["trajectory"][idx_frame-stride:idx_frame] = traj[idx_frame-stride:idx_frame]
    idx_frame += stride

if idx_frame != n_frames:
    traj_h5["trajectory"][idx_frame-stride:] = traj[idx_frame-stride:]

traj_h5.close()

# print("frame number 123 of filled dataset:")
# print(traj_h5["trajectory"][123], "\n")

# print("Last frame")
# print(traj_h5["trajectory"][-1], "\n")


th = time()
traj_h5 = h5py.File(fname_traj, "r")
traj_2 = np.array(traj_h5["trajectory"])
th = time() - th

ths = time()
traj_h5 = h5py.File(fname_traj, "r")
traj_2_sliced = np.array(traj_h5["trajectory"][100:200])
ths = time() - ths
traj_3 = traj_h5["trajectory"][:]

print(type(traj))
print(type(traj_2))
print(type(traj_h5))
print(type(traj_3))
print(traj_3.shape)


wrong_frames = 0
for frame1, frame2 in zip(traj, traj_2):
    if not np.allclose(frame1, frame2):
        wrong_frames += 1
print("Number of wrong frames:", wrong_frames, "\n")
print(f"Time to load trajectory (old):  {tt:.5f}")
print(f"Time to load trajectory (h5):   {th:.5f}\n")
print(f"Time to load sliced trajectory (old): {ttso:.5f}")
print(f"Time to load sliced trajectory (h5):  {ths:.5f}\n")

ts1 = time()
mc.utils.save_trajectory(cfg, traj, fname_traj)
ts1 = time() - ts1

ts2 = time()
traj_h5 = h5py.File(fname_traj, "w")
traj_h5.create_dataset("trajectory", traj_shape, dtype="float64")
traj_h5["trajectory"][:] = traj_2
ts2 = time() - ts2

print(f"Time to save trajectory (old):  {ts1:.5f}")
print(f"Time to save trajectory (h5):   {ts2:.5f}\n")