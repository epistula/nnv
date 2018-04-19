import argparse
import subprocess
import pdb
import glob

def run(cmd):
    subprocess.check_call(cmd, shell=True)


list_of_files = [
				# ('datasets/lsun_celebA_data/celebA_64_cubic/splits/train/', 'train.npy'),
				# ('datasets/lsun_celebA_data/celebA_64_cubic/splits/test/', 'test.npy'),
				('datasets/cifar10_data/cifar-10-batches-py/', 'regulartest_data.npy'),
				('datasets/cifar10_data/cifar-10-batches-py/', 'regulartest_label.npy'),
				('datasets/cifar10_data/cifar-10-batches-py/', 'regulartrain_data.npy'),
				('datasets/cifar10_data/cifar-10-batches-py/', 'regulartrain_label.npy'),
]

main_folder = '~/'
mac_loc_str = '/Users/MeVlana/'

# cirra_loc_str = 'instance-central1c-0cpu2mem50hd-hour0-022-month15-80:'+main_folder
# zone = 'us-central1-c'

# cirra_loc_str = 'instance1-east1d-1gpu2cpu12mem50hd-hour0-402-month293-12'
cirra_loc_str = 'instance1-east1d-1gpu1cpu6mem50hd-hour0-361-month263-67'
zone = 'us-east1-d'

for f in list_of_files:
	run('echo gcloud compute scp '+ mac_loc_str + f[0]+ f[1] +' '+cirra_loc_str+':'+main_folder +f[0]+' '+ '--zone ' + zone)
	run('gcloud compute scp '+ mac_loc_str + f[0]+ f[1] +' '+cirra_loc_str+':'+main_folder +f[0]+' '+ '--zone ' + zone)


# include_str_mac = ''
# for i in range(len(list_of_folders)):
# 	cirra_loc_str = cirra_address+ ':' + main_folder+list_of_folders[i]
# 	run('echo rsync -avzCL  --progress --prune-empty-dirs '+ cirra_loc_str + ' ' + mac_loc_str)
# 	run('rsync -avzCL  --progress --prune-empty-dirs '+ cirra_loc_str + ' ' + mac_loc_str)

