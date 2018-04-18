import argparse
import subprocess
import pdb

def run(cmd):
    subprocess.check_call(cmd, shell=True)

# folder_to_sync = 'experiments-SecondDetWGANGP-FLOWERS/fbc15382963644319d94ca4531c482b2/' 
folder_to_sync = 'EEEexperimentsLast-PDWGANCannon2-CIFAR10/35ecda24a3bf465887917436e4e7131f/'

def maintensorflow():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--action', type=str, default='mac')
	parser.add_argument('-m', '--main_only', type=bool, default=False)
	parser.add_argument('-c', '--cirra_address', type=str, default='104.155.190.99')
	parser.add_argument('-py', '--python_files', type=bool, default=False)
	parser.add_argument('--include_ext', type=list, default=[])

	parser.add_argument('--include_sync_to_mac', type=list, default=[".py", ".th", ".png", ".jpg", ".jpeg", ".txt", ".text"])#, "/checkpoint2/*"])
	parser.add_argument('--mac_loc', type=str, default='/Users/MeVlana/tensorflowCode/oia-master/tensorflow/')
	parser.add_argument('--cirra_loc', type=str, default='/Users/MeVlana/tensorflowCode/')
	parser.add_argument('--folder_to_sync', type=str, default='/Users/MeVlana/tensorflowCode/')
	args = parser.parse_args()
	
	if args.main_only:
		for i, e in enumerate(args.include_sync_to_mac):
			if e in [".png", ".jpg", ".jpeg"]:
				args.include_sync_to_mac[i] = "m"+e

	mac_loc_str = ' ' + args.mac_loc + folder_to_sync
	cirra_loc_str = ' ' + args.cirra_address + ':' + args.cirra_loc + folder_to_sync
	
	include_str_mac = ''
	if args.action == 'mac':
		for i in range(len(args.include_sync_to_mac)):
			include_str_mac = include_str_mac + ' --include="*'+args.include_sync_to_mac[i]+'"'
	if args.python_files:
		include_str_mac = include_str_mac + ' --include="*.py"'

	# include_str_cirra = ''
	# if args.action == 'cirra':
	# 	for i in range(len(args.include_sync_to_cirra)):
	# 		include_str_cirra = include_str_cirra + ' --include="*'+args.include_sync_to_cirra[i]+'"'

	include_train = False
	if args.action == 'mac':
		if include_train:
			run('echo rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
			run('rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
		else:
			# pdb.set_trace()
			run('echo rsync -avzCL  --progress --prune-empty-dirs --exclude="*Train*" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
			run('rsync -avzCL  --progress --prune-empty-dirs --exclude="*Train*" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)

	# elif args.action == 'cirra':
	# 	run('echo rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	# 	run('rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	else:
		raise Exception('invalid action')


maintensorflow()





