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
# cirra_loc_str = 'instance1-east1d-1gpu1cpu6mem50hd-hour0-361-month263-67'
cirra_loc_str = 'instance-east1d-8cpu15mem30hd-hour0-234-month170-89'
zone = 'us-east1-d'


# list_of_files = [('Visualization/test_TSNE_prior_posterior/', '.png'),

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

# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/a8455ec856274c7aa3cd4116a5dcdb94/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/43905b65ee8a4a269bf4554d7f957b67/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/87c75aa807444307813db5de5184d7ed/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/a0d5a891c0464bdea3be8abc6c84c3bb/'

# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/431df31f91944565b6b7698a971f7577/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/16623e0803a342df8172c542e8279748/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/164e400217cb483a8c4cc1af0e0c0c3a/'
# experiment_folder = './EEEexperimentsLastMIX-PDWGANCannon2-CIFAR10/98ee5405af7a4f64ab2570dc128e9237/'


experiment_folders = [
					'./EEEexperimentsStable-WAEVanilla-INTENSITY/0156f7e3571449d384b8b465b581fbd5/', #Gaussian
					'./EEEexperimentsStable-WAEVanilla-INTENSITY/c762a8e29481410ca7ad09d859d94e5e/', #Uniform
					'./EEEexperimentsStable-WAEVanilla-INTENSITY/b09ffa0646964cd79da8d126754c67d8/', #Uniform rand
					 ]
for experiment_folder in experiment_folders:
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

