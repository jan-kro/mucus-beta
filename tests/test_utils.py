from mucus_rust.utils import load_trajectory
from mucus_rust.config import Config


cfg_path = "/net/storage/janmak98/masterthesis/output/connected-S-mesh-charged/configs/cfg_connected-S-mesh-charged_v0.toml"

# load the configuration
cfg = Config.from_toml(cfg_path)
traj = load_trajectory(cfg, frame_range=(3, 5))
print(traj.shape)
