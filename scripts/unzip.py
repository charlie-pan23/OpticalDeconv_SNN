import tarfile
import os

archive_path = r"/datasets/ibmGestureTest.tar..gz"
extract_path = r"D:\ProgramProject_Hub\OpticalDeconv_SNN\datasets\DVSGesture"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

print(f"Unpacking {archive_path} ...")
try:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("\nUnpacked successfully! Check the files in datasets/DVSGesture.")
except Exception as e:
    print(f"\nFailed: {e}")