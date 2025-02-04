# python scripts/distributions/run.py --sampler 1 --cache_size=16 --base_sampler=1 --logged&
# python scripts/distributions/run.py --sampler 2 --cache_size=16 --base_sampler=1 --logged&

# python scripts/distributions/run.py --sampler 1 --cache_size=32 --base_sampler=1 --logged&
# python scripts/distributions/run.py --sampler 2 --cache_size=32 --base_sampler=1 --logged&

# python scripts/distributions/run.py --sampler 1 --cache_size=64 --base_sampler=1 --logged&
# python scripts/distributions/run.py --sampler 2 --cache_size=64 --base_sampler=1 --logged&

# python scripts/distributions/run.py --sampler 1 --cache_size=128 --base_sampler=1 --logged&
# python scripts/distributions/run.py --sampler 2 --cache_size=128 --base_sampler=1 --logged&

# python scripts/distributions/run.py --sampler 1 --base_sampler=1 --logged&
# python scripts/distributions/run.py --sampler 2 --base_sampler=1 --logged&


# for i in {1..6}; 
# do 
    # python scripts/distributions/run.py --sampler $i --logged --alg 1 &
    # python scripts/distributions/run.py --sampler $i --logged --alg 2 & 
    # python scripts/distributions/run.py --sampler $i --logged --alg 3 &
    # python scripts/distributions/run.py --sampler $i --logged --alg 4 &
    # python scripts/distributions/run.py --sampler $i --logged --alg 5 &
    # python scripts/distributions/run.py --sampler $i --logged --alg 6 & 
# done 

# python scripts/distributions/run.py --logged --alg 1 --sampler 7
# python scripts/distributions/run.py --logged --alg 4 --sampler 7

python scripts/distributions/run.py --logged --alg 3 --sampler 7