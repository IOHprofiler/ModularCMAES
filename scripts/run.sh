
for sampler in 0 1 2;
do 
    for cache_size in 0 16 32 64 128;
    do
        python run.py --logged --sampler=$sampler --cache_size=$cache_size "$@" &
    done
done