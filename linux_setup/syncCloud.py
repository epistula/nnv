import argparse
import subprocess
import pdb
import glob

def run(cmd):
    subprocess.check_call(cmd, shell=True)

def maintensorflow():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--action', type=str, default='cirra', choices=['both', 'cirra', 'mac'])
	parser.add_argument('-m', '--main_only', type=bool, default=False)
	parser.add_argument('-c', '--cirra_address', type=str, default='104.155.190.99')
	parser.add_argument('-py', '--python_files', type=bool, default=False)
	parser.add_argument('--include_ext', type=list, default=[])

	# parser.add_argument('--include_sync_to_mac', type=list, default=[".th", ".png", ".jpg", ".jpeg", ".txt", ".text", ".meta", ".index", ".data*", "checkpoint2/*"])
	# parser.add_argument('--include_sync_to_mac', type=list, default=[".py", ".th", ".png", ".jpg", ".jpeg", ".txt", ".text"])#, "/checkpoint2/*"])
	parser.add_argument('--include_sync_to_mac', type=list, default=[".py", ".txt", ".text"])#, "/checkpoint2/*"])
	# parser.add_argument('--include_sync_to_mac', type=list, default=[".th", ".png", ".jpg", ".jpeg", ".txt", ".text", "/checkpoint2/*"])
	# parser.add_argument('--include_sync_to_cirra', type=list, default=[".tgz", ".py", ".sh", ".gz", "NiceRun/*/*/*/.png", ".mat"])
	parser.add_argument('--include_sync_to_cirra', type=list, default=[".tgz", ".py", ".sh", ".gz"])
	# parser.add_argument('--include_sync_to_cirra', type=list, default=[".py", ".sh", ".tgz"])

	# parser.add_argument('--mac_loc', type=str, default='/Users/mevlana/ocean/dataset/dataset_both2/scripted/')
	# parser.add_argument('--cirra_loc', type=str, default='mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/dataset/dataset_both2/scripted/')
	parser.add_argument('--mac_loc', type=str, default='/Users/MeVlana/tensorflowCode/oia-master/tensorflow/')
	# parser.add_argument('--cirra_loc', type=str, default='/Users/MeVlana/tensorflowCode/')
	parser.add_argument('--cirra_loc', type=str, default='~/tensorflowCode/')
	args = parser.parse_args()

	if args.main_only:
		for i, e in enumerate(args.include_sync_to_mac):
			if e in [".png", ".jpg", ".jpeg"]:
				args.include_sync_to_mac[i] = "m"+e

	mac_loc_str = ' ' + args.mac_loc
	cirra_loc_str = ' ' + args.cirra_address + ':' + args.cirra_loc
	
	include_str_mac = ''
	if args.action == 'mac':
		for i in range(len(args.include_sync_to_mac)):
			include_str_mac = include_str_mac + ' --include="*'+args.include_sync_to_mac[i]+'"'
	if args.python_files:
		include_str_mac = include_str_mac + ' --include="*.py"'

	include_str_cirra = ''
	cirra_file_list = []
	if args.action == 'cirra':
		for i in range(len(args.include_sync_to_cirra)):
			# include_str_cirra = include_str_cirra + ' --include="*'+args.include_sync_to_cirra[i]+'"'
			# cirra_file_list.append(glob.glob(mac_loc_str[1:]+'**/*'+args.include_sync_to_cirra[i], recursive=True))
			cirra_file_list=cirra_file_list+(glob.glob(mac_loc_str[1:]+'*'+args.include_sync_to_cirra[i]))
			cirra_file_list=cirra_file_list+(glob.glob(mac_loc_str[1:]+'datasetLoaders/*'+args.include_sync_to_cirra[i]))
			cirra_file_list=cirra_file_list+(glob.glob(mac_loc_str[1:]+'linux_setup/*'+args.include_sync_to_cirra[i]))
			cirra_file_list=cirra_file_list+(glob.glob(mac_loc_str[1:]+'models/**/*'+args.include_sync_to_cirra[i], recursive=True))

		cirra_file_list_string = ''
		for i in range(len(cirra_file_list)):
			cirra_file_list_string = cirra_file_list_string +cirra_file_list[i]+' '

	if args.action == 'both':
		run('echo gsutil rsync -avzCL '+ cirra_loc_str + mac_loc_str)
		run('gsutil rsync -avzCL ' + cirra_loc_str + mac_loc_str)
		run('echo gsutil rsync -avzCL ' + mac_loc_str + cirra_loc_str)
		run('gsutil rsync -avzCL ' + mac_loc_str + cirra_loc_str)
	elif args.action == 'mac':
		# run('echo gsutil rsync -avzCL ' + cirra_loc_str + mac_loc_str)
		# run('gsutil rsync -avzCL ' + cirra_loc_str + mac_loc_str)
		run('gcloud compute scp -v '+ cirra_file_list_string +' '+cirra_loc_str+' '+ '--zone us-central1-c')
	elif args.action == 'cirra':
		# run('echo gsutil rsync -avzCL ' + mac_loc_str + cirra_loc_str)
		# run('gsutil rsync -avzCL ' + mac_loc_str + cirra_loc_str)
		# pdb.set_trace()
		run('gcloud compute scp '+ cirra_file_list[0] +' '+cirra_loc_str+' '+ '--zone us-central1-c --recurse')
		# run('gcloud compute scp -v '+ cirra_file_list_string +' '+cirra_loc_str+' '+ '--zone us-central1-c')
	else:
		raise Exception('invalid action')

	# gcloud compute scp /Users/MeVlana/tensorflowCode/oia-master/tensorflow/gan_*.py instance-central1c-0cpu2mem50hd-hour0-022-month15-80:~/ --zone us-central1-c

	# if args.action == 'both':
	# 	run('echo gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
	# 	run('gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
	# 	run('echo gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	# 	run('gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	# elif args.action == 'mac':
	# 	run('echo gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
	# 	run('gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
	# elif args.action == 'cirra':
	# 	run('echo gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	# 	run('gsutil rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync*.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	# else:
	# 	raise Exception('invalid action')

def mainpytorch():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--action', type=str, default='cirra', choices=['both', 'cirra', 'mac'])
	parser.add_argument('-c', '--cirra_address', type=str, default='104.155.190.99')
	parser.add_argument('-py', '--python_files', type=bool, default=False)
	parser.add_argument('--include_ext', type=list, default=[])

	# parser.add_argument('--include_sync_to_mac', type=list, default=[".th", ".png", ".jpg", ".jpeg", ".txt", ".text", ".meta", ".index", ".data*", "checkpoint2/*"])
	parser.add_argument('--include_sync_to_mac', type=list, default=[".py", ".th", ".png", ".jpg", ".jpeg", ".txt", ".text"])#, "/checkpoint2/*"])
	# parser.add_argument('--include_sync_to_mac', type=list, default=[".th", ".png", ".jpg", ".jpeg", ".txt", ".text", "/checkpoint2/*"])
	parser.add_argument('--include_sync_to_cirra', type=list, default=[".tgz", ".py", ".sh", ".gz", "NiceRun/*/*/*/.png", ".mat"])
	# parser.add_argument('--include_sync_to_cirra', type=list, default=[".py", ".sh", ".tgz"])

	# parser.add_argument('--mac_loc', type=str, default='/Users/mevlana/ocean/dataset/dataset_both2/scripted/')
	# parser.add_argument('--cirra_loc', type=str, default='mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/dataset/dataset_both2/scripted/')
	parser.add_argument('--mac_loc', type=str, default='/Users/MeVlana/tensorflowCode/oia-master/pytorch/')
	parser.add_argument('--cirra_loc', type=str, default='/home/mcgemici/pytorchCode/')
	args = parser.parse_args()

	mac_loc_str = ' ' + args.mac_loc
	cirra_loc_str = ' ' + args.cirra_address + ':' + args.cirra_loc
	
	include_str_mac = ''
	if args.action == 'mac':
		for i in range(len(args.include_sync_to_mac)):
			include_str_mac = include_str_mac + ' --include="*'+args.include_sync_to_mac[i]+'"'
	if args.python_files:
		include_str_mac = include_str_mac + ' --include="*.py"'

	include_str_cirra = ''
	if args.action == 'cirra':
		for i in range(len(args.include_sync_to_cirra)):
			include_str_cirra = include_str_cirra + ' --include="*'+args.include_sync_to_cirra[i]+'"'


	if args.action == 'both':
		run('echo rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
		run('rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
		run('echo rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
		run('rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	elif args.action == 'mac':
		run('echo rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
		run('rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
	elif args.action == 'cirra':
		run('echo rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
		run('rsync -avzCL  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	else:
		raise Exception('invalid action')

maintensorflow()
# mainpytorch()










































# parser.add_argument('exclude_ext', type=list, default=["th", "blah", "zip", "tar", "gz", "md", "json", "pb"])
# find . -type d -empty -print
# MAC:
# scp -r /Users/mevlana/pytorch/ocean/dataset_3/* mevlana@1.cirrascale.sci.openai-tech.com:/home/mevlana/research/pytorch/ocean/dataset_3/

# rsync -avzC --filter='-rs_*/.svn*' --include="*/" --include='*.py'  --exclude="*" /Users/mevlana/pytorch/ mevlana@0.cirrascale.sci.openai.org:/home/mevlana/research/pytorch/
# rsync -avzC --filter='-rs_*/.svn*' --include="*/" --include="pretrained/*"  --include='*.py'  --exclude="*" /Users/mevlana/research/openai/pixel-cnn/ mevlana@3.cirrascale.sci.openai.org:/home/mevlana/research/pixel-cnn/

# LINUX: Run from Mac
# rsync -avzC --filter='-rs_*/.svn*' --include="*/" --include="*/*/" --include='*.py' --include='*.png'  --include='*.txt' --exclude="*" mevlana@0.cirrascale.sci.openai.org:/home/mevlana/research/pytorch/ /Users/mevlana/pytorchCirra/ 

# rm /Users/mevlana/research/openai/pixel-cnn/results/save/*
# rsync -avzC --filter='-rs_*/.svn*' --include="*/" --include="*/*/" --include='*.py' --include='*.png'  --include='*.txt' --exclude="*" mevlana@3.cirrascale.sci.openai.org:/tmp/pxpp/mevsave2  /Users/mevlana/research/openai/pixel-cnn/results/ 

# ssh 3.cirrascale.sci.openai.org
# ssh 4.cirrascale.sci.openai.org
# cirrascale-cli reserve -g 8 -t 1d
# source activate tensorflowGPU
# cd research/pixel-cnn/
# screen -R pixel
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --nr_gpu 8

# rsync -avzC --filter='-rs_*/.svn*' --include="*/"  --include='*.py'   --include='*.txt'  --include='*.jpg' --include='*.png'  --include='*.th' --exclude="*" mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/pytorch/ /Users/mevlana/pytorch/ 
# rsync -avzC --filter='-rs_*/.svn*' --include="*/"  --include='*.py'   --include='*.txt'  --include='*.jpg' --include='*.png'  --exclude="*" /Users/mevlana/ocean/ mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/ 
# ls | wc -l

# rsync -avzC --filter='-rs_*/.svn*'  --include='*.py'   --include='*.txt' exclude="*" /Users/mevlana/ocean/ mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/ 

# scp -r /Users/mevlana/ocean/*.py mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/
# rsync -avzhe --progress /Users/mevlana/ocean/*.py mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/

# rsync -avzC  --progress --exclude="*.th" --exclude="*/*"        \
# /Users/mevlana/ocean/pytorch/           \
# mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/pytorch/

# rsync -avzC  --progress --exclude="*.th" --exclude="*/*"        \
# /Users/mevlana/ocean/pytorch/       \
# mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/pytorch/  

# rsync -avzC  --progress        \
# mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/pytorch/experiments8/e3cb32030f9d404f829ace744705386e/       \
# /Users/mevlana/ocean/pytorch/experiments8/e3cb32030f9d404f829ace744705386e/       
