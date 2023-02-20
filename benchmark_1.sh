echo 'BENCHMARK START'
for i in {1..15}
do
    echo iter $i
    python main.py -pid 3 -iid $i -d 5 -pt 1  -n 'ModCMA'
    python main.py -pid 3 -iid $i -d 20 -pt 1  -n 'ModCMA'
    python main.py -pid 3 -iid $i -d 5 -pt 2 -n 'SP-CMA'
    python main.py -pid 3 -iid $i -d 20 -pt 2 -n 'SP-CMA'
    python main.py -pid 4 -iid $i -d 5 -pt 1  -n 'ModCMA'
    python main.py -pid 4 -iid $i -d 20 -pt 1  -n 'ModCMA'
    python main.py -pid 4 -iid $i -d 5 -pt 2 -n 'SP-CMA'
    python main.py -pid 4 -iid $i -d 20 -pt 2 -n 'SP-CMA'
done
echo 'BENCHMARK COMPLETE'