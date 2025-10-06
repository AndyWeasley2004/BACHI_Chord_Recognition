import json
import os
import shutil

root_dir = '../data_root/pop909_collection'
test_files = json.load(open('pop909_test.json', 'r'))
test_dir = './pop909_test'
os.makedirs(test_dir, exist_ok=True)

for file in test_files:
    destination_path = os.path.join(test_dir, file)
    shutil.copytree(os.path.join(root_dir, file), destination_path, dirs_exist_ok=True)