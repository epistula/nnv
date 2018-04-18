import argparse
import subprocess
import pdb

def run(cmd):
    subprocess.check_call(cmd, shell=True)


list_of_folders = [
'imagenet_data/imagenet_64/train',
'imagenet_data/imagenet_64/test',
'cub_data',
'flowers_data',
'cat_data',
'lsun_celebA_data',
'cifar10_data',
]

main_folder = '/home/mcgemici/datasets/'
mac_loc_str = '/Users/MeVlana/datasets/'
cirra_address = 'mcgemici@146.50.28.200'

include_str_mac = ''
for i in range(len(list_of_folders)):
	cirra_loc_str = cirra_address+ ':' + main_folder+list_of_folders[i]
	run('echo rsync -avzCL  --progress --prune-empty-dirs '+ cirra_loc_str + ' ' + mac_loc_str)
	run('rsync -avzCL  --progress --prune-empty-dirs '+ cirra_loc_str + ' ' + mac_loc_str)

