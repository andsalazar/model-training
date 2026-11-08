import glob, json, os
import numpy as np, imageio.v3 as iio
paths = sorted(glob.glob("/local/adapter5s_v2/eval/*.mp4")) + sorted(glob.glob("/local/pilot5s/teacher/*_s100.mp4")) + ["/local/pilot5s/teacher/river_s200.mp4", "/local/pilot5s/teacher/river_s300.mp4"]
for p in paths:
    f = np.asarray(iio.imread(p, plugin="FFMPEG"))[..., :3].astype(np.float32)
    luma = (f[..., 0] * .299 + f[..., 1] * .587 + f[..., 2] * .114).mean(axis=(1, 2))
    j = np.abs(np.diff(luma))
    print(f"{p.split('/')[-2][:7]}/{os.path.basename(p):<34} max_jump {j.max():6.2f} at frame {int(j.argmax()):>3}  last8 max {j[-8:].max():6.2f}  body median {np.median(j[:-8]):.3f}")