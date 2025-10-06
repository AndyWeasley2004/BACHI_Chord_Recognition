import os
import subprocess

root = '../classical_all/'
subdirs = os.listdir(root)
file_paths = [os.path.join(root, subdir, subdir + '.mid') for subdir in subdirs]

for path in file_paths:
    result = subprocess.run(['bash', 'preprocessing.sh', path], check=True)
