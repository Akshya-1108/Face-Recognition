import os
import subprocess

# Set the path for LabelImg
LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

# Create the directory if it doesn't exist
if not os.path.exists(LABELIMG_PATH):
    os.makedirs(LABELIMG_PATH)
    subprocess.run(["git", "clone", "https://github.com/tzutalin/labelImg", LABELIMG_PATH])

# Build LabelImg
if os.name == 'posix':
    subprocess.run(["make", "qt5py3"], cwd=LABELIMG_PATH)
elif os.name == 'nt':
    subprocess.run(["pyrcc5", "-o", "libs/resources.py", "resources.qrc"], cwd=LABELIMG_PATH)

# Run LabelImg
subprocess.run(["python", "labelImg.py"], cwd=LABELIMG_PATH)
