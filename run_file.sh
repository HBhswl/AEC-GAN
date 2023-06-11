#!/bash/bin

# train the model
CUDA_VISIBLE_DEVICES=0 python train.py -datasets 'etth1'     -base_dir 'results/p168_q336' -p 168 -q 336 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 
CUDA_VISIBLE_DEVICES=1 python train.py -datasets 'etth2'     -base_dir 'results/p168_q336' -p 168 -q 336 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 
CUDA_VISIBLE_DEVICES=2 python train.py -datasets 'ettm1'     -base_dir 'results/p96_q192'  -p 96  -q 192 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 
CUDA_VISIBLE_DEVICES=3 python train.py -datasets 'ettm2'     -base_dir 'results/p96_q192'  -p 96  -q 192 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 
CUDA_VISIBLE_DEVICES=4 python train.py -datasets 'us_births' -base_dir 'results/p168_q336' -p 168 -q 336 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 
CUDA_VISIBLE_DEVICES=5 python train.py -datasets 'ILI'       -base_dir 'results/p18_q36'   -p 18  -q 36  -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 

# generate data
CUDA_VISIBLE_DEVICES=0 python train.py -datasets 'etth1'     -base_dir 'results/p168_q336' -p 168 -q 336 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 -test 
CUDA_VISIBLE_DEVICES=1 python train.py -datasets 'etth2'     -base_dir 'results/p168_q336' -p 168 -q 336 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 -test 
CUDA_VISIBLE_DEVICES=2 python train.py -datasets 'ettm1'     -base_dir 'results/p96_q192'  -p 96  -q 192 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 -test 
CUDA_VISIBLE_DEVICES=3 python train.py -datasets 'ettm2'     -base_dir 'results/p96_q192'  -p 96  -q 192 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 -test 
CUDA_VISIBLE_DEVICES=4 python train.py -datasets 'us_births' -base_dir 'results/p168_q336' -p 168 -q 336 -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 -test 
CUDA_VISIBLE_DEVICES=5 python train.py -datasets 'ILI'       -base_dir 'results/p18_q36'   -p 18  -q 36  -use_cuda -algos 'AECGAN' -total_steps 10000 -batch_size 200 -noise_type min_adv -use_ec 2 -test 

