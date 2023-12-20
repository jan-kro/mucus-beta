import h5py
import mucus_rust as mc
from time import time
import numpy as np
import os

# print(" ~~~ TEST H5 WRITING AND LOADING ~~~\n")

# data = np.random.randint(0, 100, size=(2, 5, 3))
# print("data:")
# print(data, "\n")

# fname = "/home/janmak98/mucus_rust_test/tests/testwrite.hdf5"
# print("fname:", fname, "\n")

# # delete file if it exists
# if os.path.exists(fname):
#     os.remove(fname)

# print("open h5 with a mode and initialize shape (100, 5, 3) (file doesnt exist yet)")
# h5 = h5py.File(fname, "a")
# print("create dataset")
# h5.create_dataset("test", (100, 5, 3), dtype="float64")
# print("write data to dataset")
# h5["test"][:2] = data
# print("close h5")
# h5.close()

# print("open h5 with a mode (file exists)")
# h5 = h5py.File(fname, "a")
# print("read data from dataset")
# data2 = np.array(h5["test"])
# print("data2:")
# print(data2, "\n")

# print("close h5")
# h5.close()

# print("create new data")
# data3 = np.random.randint(0, 100, size=(2, 5, 3))
# print("data3:")
# print(data3, "\n")


# print("open h5 with a mode (file exists)")
# h5 = h5py.File(fname, "a")
# print("write data to dataset")
# h5["test"][:2] = data3

# print("read data from dataset")
# data4 = np.array(h5["test"])
# print("data4:")
# print(data4, "\n")

# print("add data to dataset")
# h5["test"][2:4] = data

# print("read data from dataset")
# data5 = np.array(h5["test"])
# print("data5:")
# print(data5, "\n")

# print("close h5")
# h5.close()

# print("open h5 with r+ mode (file exists)")
# h5 = h5py.File(fname, "r+")
# print("add data to dataset")
# h5["test"][2:4] = data
# print("data6:")
# print(np.array(h5["test"]), "\n")

# print("close h5")
# h5.close()




print(" ~~~ TEST SLICED WRITING AND LOADING ~~~\n")

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
traj_shape = (n_frames, cfg.n_particles, 3)
# print("Shape of trajectory (calculated):")
# print(traj_shape, "\n")
traj_h5 = h5py.File(fname_traj, "w")
traj_h5.create_dataset("trajectory", traj_shape, dtype="float64")


# print("frame number 123 of created dataset:")
# print(traj_h5["trajectory"][123], "\n")

stride = 93
idx_chunk = 1
idx_traj = 0
chunksize = 30

traj_chunk = np.zeros((chunksize, cfg.n_particles, 3))
traj_chunk[0] = traj[0]

for step in range(1, n_frames*stride):
    if step % stride == 0:
        traj_chunk[idx_chunk] = traj[int(step/stride)]
        idx_chunk += 1
    if idx_chunk == chunksize:
        traj_h5["trajectory"][idx_traj:idx_traj+chunksize] = traj_chunk
        idx_traj += chunksize
        idx_chunk = 0
        
print(idx_traj)
print(n_frames)
if idx_traj!= n_frames:
    traj_h5["trajectory"][idx_traj:] = traj_chunk[:idx_chunk]

# idx_frame = 1*stride
# while idx_frame < n_frames:
#     traj_h5["trajectory"][idx_frame-stride:idx_frame] = traj[idx_frame-stride:idx_frame]
#     idx_frame += stride

# if idx_frame != n_frames:
#     traj_h5["trajectory"][idx_frame-stride:] = traj[idx_frame-stride:]

traj_h5.close()

# print("frame number 123 of filled dataset:")
# print(traj_h5["trajectory"][123], "\n")

# print("Last frame")
# print(traj_h5["trajectory"][-1], "\n")


th = time()
traj_h5 = h5py.File(fname_traj, "r")
traj_2 = np.array(traj_h5["trajectory"])
th = time() - th
traj_h5.close()

# print("traj:")
# print(traj[4])
# print("traj_2:")
# print(traj_2[4])
# print('traj end:')
# print(traj[-1])
# print('traj_2 end:')
# print(traj_2[-1])


ths = time()
traj_h5 = h5py.File(fname_traj, "r")
traj_2_sliced = np.array(traj_h5["trajectory"][100:200])
ths = time() - ths

th3 = time()
traj_h5 = h5py.File(fname_traj, "r") 
traj_3 = traj_h5["trajectory"][:]
th3 = time() - th3

# print(type(traj))
# print(type(traj_2))
# print(type(traj_h5))
# print(type(traj_3))
# print(traj_3.shape)


wrong_frames = 0
for frame1, frame2 in zip(traj, traj_2):
    if not np.allclose(frame1, frame2):
        wrong_frames += 1
print("Number of wrong frames:", wrong_frames, "\n")
print(f"Time to load trajectory (old):  {tt:.5f}")
print(f"Time to load trajectory (h5):   {th:.5f}")
print(f"Time to load trajectory (h5 2): {th3:.5f}")
print(f"Speedup (old/h5):               {(tt/th)*100:.2f} %\n")
print(f"Speedup (old/h5 2):             {(tt/th3)*100:.2f} %\n")

print(f"Time to load sliced trajectory (old): {ttso:.5f}")
print(f"Time to load sliced trajectory (h5):  {ths:.5f}")
print(f"Speedup (old/h5):                     {(ttso/ths)*100:.2f} %\n")
ts1 = time()
#mc.utils.save_trajectory(cfg, traj, fname_traj)
ts1 = time() - ts1
 
traj_h5.close()

ts2 = time()
traj_h5 = h5py.File(fname_traj, "w")
traj_h5.create_dataset("trajectory", traj_shape, dtype="float64")
traj_h5["trajectory"][:] = traj_2
ts2 = time() - ts2

print(f"Time to save trajectory (old):  {ts1:.5f}")
print(f"Time to save trajectory (h5):   {ts2:.5f}")
print(f"Speedup (old/h5):               {(ts1/ts2)*100:.2f} %\n")

print(" ~~~ TEST GET PATH ~~~\n")

ptraj = mc.utils.get_path(cfg)
print("old:", ptraj)

htraj = mc.utils.get_path(cfg, filetype="results")
print("new:", htraj, "\n")

print(" ~~~ TEST GET RESULTS ~~~\n")

print('Create trajectory file ...')
fname_results =  mc.utils.get_path(cfg, filetype="results") # , dataset="trajectory", shape=traj.shape)
if os.path.exists(fname_results):
    os.remove(fname_results)
fname_results =  mc.utils.get_path(cfg, filetype="results") # , dataset="trajectory", shape=traj.shape)
h5_results = h5py.File(fname_results, "a")
print('keys:', h5_results.keys())
print('type:', type(h5_results), "\n")
print('create trajectory dataset...')
h5_results.create_dataset("trajectory", traj_shape, dtype="float64")
print("close file...")
h5_results.close()

print('open file again...')
fname_results =  mc.utils.get_path(cfg, filetype="results") # , dataset="trajectory", shape=traj.shape)
h5_results = h5py.File(fname_results, "a")
print('keys:', h5_results.keys())
print('type:', type(h5_results))
print('h5 traj shape:', h5_results["trajectory"].shape)
print("h5 traj type:", type(h5_results["trajectory"]))
print("traj shape:", traj.shape, "\n")

print('load frames into trajectory...')
h5_results["trajectory"][:] = traj
print('shape:', h5_results["trajectory"].shape)
print('close file...\n')
h5_results.close()


print("load dataset 'trajectory' again...")
fname_results_2 = mc.utils.get_path(cfg, filetype="results")
h5_results_2 = h5py.File(fname_results_2, "a")
print('shape:', h5_results_2["trajectory"].shape)

print("traj and trajh5 are the same: " , np.all(traj==h5_results_2["trajectory"]))

print('close file...\n')
h5_results_2.close()