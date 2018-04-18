import argparse
import subprocess

def run(cmd):
    subprocess.check_call(cmd, shell=True)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--action', type=str, default='cirra', choices=['both', 'cirra', 'mac'])
	parser.add_argument('-c', '--cirra_address', type=str, default='104.155.190.99')
	parser.add_argument('-py', '--add_python_extension', type=bool, default=False)
	parser.add_argument('--include_ext', type=list, default=[])

	# parser.add_argument('--include_sync_to_mac', type=list, default=[".th", ".png", ".jpg", ".jpeg", ".txt", ".text", ".meta", ".index", ".data*", "checkpoint2/*"])
	parser.add_argument('--include_sync_to_mac', type=list, default=[".th", ".png", ".jpg", ".jpeg", ".txt", ".text", "/checkpoint2/*"])
	parser.add_argument('--include_sync_to_cirra', type=list, default=[".py", ".sh"])
	# parser.add_argument('--include_sync_to_cirra', type=list, default=[".py", ".sh", ".tgz"])

	# parser.add_argument('--mac_loc', type=str, default='/Users/mevlana/ocean/dataset/dataset_both2/scripted/')
	# parser.add_argument('--cirra_loc', type=str, default='mevlana@4.cirrascale.sci.openai.org:/home/mevlana/research/ocean/dataset/dataset_both2/scripted/')
	parser.add_argument('--mac_loc', type=str, default='/Users/MeVlana/Research/tensorflow/')
	parser.add_argument('--cirra_loc', type=str, default='/home/mcgemici/tensorflowCode/')
	args = parser.parse_args()

	mac_loc_str = ' ' + args.mac_loc
	cirra_loc_str = ' ' + args.cirra_address + ':' + args.cirra_loc
	
	include_str_mac = ''
	if args.action == 'mac':
		for i in range(len(args.include_sync_to_mac)):
			include_str_mac = include_str_mac + ' --include="*'+args.include_sync_to_mac[i]+'"'

	# parser.add_argument('-py', '--add_python_extension', type=bool, default=False)
	if args.add_python_extension:
		include_str_mac = include_str_mac + ' --include="*.py"'

	include_str_cirra = ''
	if args.action == 'cirra':
		for i in range(len(args.include_sync_to_cirra)):
			include_str_cirra = include_str_cirra + ' --include="*'+args.include_sync_to_cirra[i]+'"'


	if args.action == 'both':
		run('echo rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
		run('rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
		run('echo rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
		run('rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	elif args.action == 'mac':
		run('echo rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
		run('rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_mac + ' --exclude="*"' + cirra_loc_str + mac_loc_str)
	elif args.action == 'cirra':
		run('echo rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
		run('rsync -avzC  --progress --prune-empty-dirs --exclude="*sync.py" --include="*/"' + include_str_cirra + ' --exclude="*"' + mac_loc_str + cirra_loc_str)
	else:
		raise Exception('invalid action')

main()










































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
