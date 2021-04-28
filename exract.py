import glob, os, time,io
import tarfile


extracted = []


files = glob.glob(
    "/home/dante0shy/dataset/nuScenes/v1.0-trainval*_blobs_lidar.*"
)
file_dict = {x.split("trainval")[-1].split("_")[0]: x for x in files}
file_list = [
    x
    for x in sorted(file_dict.items(), key=lambda x: x[0])
    if x[0] not in extracted
]
for f in file_list:
    if f[1].endswith("tgz"):
        tar = tarfile.open(f[1], "r:gz")
    else:

        tar = tarfile.open(f[1])
    print("extracting " + f[1])
    tar.extractall(path="/home/dante0shy/dataset/nuScenes")
    tar.close()
    print("finish " + f[1])

pass
