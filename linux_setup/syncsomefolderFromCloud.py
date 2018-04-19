import argparse
import subprocess
import pdb
import glob
import os

def run(cmd):
    subprocess.check_call(cmd, shell=True)

# cirra_loc_str = 'instance-central1c-0cpu2mem50hd-hour0-022-month15-80:'+main_folder
# zone = 'us-central1-c'

# cirra_loc_str = 'instance1-east1d-1gpu2cpu12mem50hd-hour0-402-month293-12'
cirra_loc_str = 'instance1-east1d-1gpu1cpu6mem50hd-hour0-361-month263-67'
zone = 'us-east1-d'

list_of_files = [('Visualization/', '.png'),
				 ('', '.txt'),
]

experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/a4682d2f5e4f4f08bdbcd2ce74e4d48c/'

main_folder = '~/tensorflowCode/'+experiment_folder
mac_loc_str = '/Users/MeVlana/CloudExperiments/'

for f in list_of_files:
	subfolder = f[0] 
	extension = f[1]
	source_dir = main_folder+subfolder
	target_dir = mac_loc_str+experiment_folder+subfolder

	if not os.path.exists(target_dir): os.makedirs(target_dir)
	run('echo gcloud compute scp '+ cirra_loc_str+':'+source_dir+'*'+extension+' '+target_dir+' '+ '--zone ' + zone)
	run('gcloud compute scp '+ cirra_loc_str+':'+source_dir+'*'+extension+' '+target_dir+' '+ '--zone ' + zone)

