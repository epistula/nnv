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

# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/a4682d2f5e4f4f08bdbcd2ce74e4d48c/'
# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/028e9730ae7d49bf8b4c8e8a8bec4b00/'
# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/f176ad31cc384e83867d8c288f42a4f2/'
# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/7dfe8226c5254d37a602a825535c639e/'
# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/1e5e17789f794ee3b58e31a295983198/'
# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/7cabbeb6985c4f10a3ec28ea502eb175/'
# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/8b51caf2f307417e804f3799f6fa3765/'
# experiment_folder = './EEEexperimentsLast-PDWGANCannon2-CIFAR10/9fd8771306aa492cb684623090977044/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/eec9b98689db41d399a4d8de7fc8b2f9/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/d42684996ab24f83978f7bb65c7bc9b4/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/9ad261a6052b43bea6ada39a624711f6/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/c8d9f2e6f41f452d88e2268f2e3dc69a/'
experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/a8455ec856274c7aa3cd4116a5dcdb94/'

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

