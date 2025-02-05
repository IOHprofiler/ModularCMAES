python run.py --sampler 1 --cache_size=16 --base_sampler=1 --logged&
python run.py --sampler 2 --cache_size=16 --base_sampler=1 --logged&

python run.py --sampler 1 --cache_size=32 --base_sampler=1 --logged&
python run.py --sampler 2 --cache_size=32 --base_sampler=1 --logged&

python run.py --sampler 1 --cache_size=64 --base_sampler=1 --logged&
python run.py --sampler 2 --cache_size=64 --base_sampler=1 --logged&

python run.py --sampler 1 --cache_size=128 --base_sampler=1 --logged&
python run.py --sampler 2 --cache_size=128 --base_sampler=1 --logged&

python run.py --sampler 1 --base_sampler=1 --logged&
python run.py --sampler 2 --base_sampler=1 --logged&


for i in {1..6}; 
do 
    for j in {1..6}; 
    do 
        python run.py --sampler $i --logged --alg &j &
    done
done 
