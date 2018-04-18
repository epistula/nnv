import argparse
import subprocess
import pdb
import glob

list_commands = [
				 'mkdir ~/datasets/',
				 # 'mkdir ~/datasets/cat_data',
				 'mkdir ~/datasets/cifar10_data',
				 'mkdir ~/datasets/cifar10_data/cifar-10-batches-py',
				 # 'mkdir ~/datasets/cub_data',
				 # 'mkdir ~/datasets/flowers_data',
				 'mkdir ~/datasets/lsun_celebA_data',
				 'mkdir ~/datasets/lsun_celebA_data/celebA_64_cubic',
				 'mkdir ~/datasets/lsun_celebA_data/celebA_64_cubic/splits',
				 'mkdir ~/datasets/lsun_celebA_data/celebA_64_cubic/splits/train',
				 'mkdir ~/datasets/lsun_celebA_data/celebA_64_cubic/splits/test',
]

for command in list_commands:
	run('echo ' + command)
	run(command)











































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
