

python scripts/distributions/run.py --sampler 1 --cache_size=32 --base_sampler=1 --logged&
python scripts/distributions/run.py --sampler 2 --cache_size=32 --base_sampler=1 --logged&

python scripts/distributions/run.py --sampler 1 --cache_size=128 --base_sampler=1 --logged&
python scripts/distributions/run.py --sampler 2 --cache_size=128 --base_sampler=1 --logged&

python scripts/distributions/run.py --sampler 1 --base_sampler=1 --logged&
python scripts/distributions/run.py --sampler 2 --base_sampler=1 --logged&


for i in {1..6}; 
do 
    python scripts/distributions/run.py --sampler $i --logged --alg 1 &
    python scripts/distributions/run.py --sampler $i --logged --alg 2 & 
    python scripts/distributions/run.py --sampler $i --logged --alg 3 & 
done 