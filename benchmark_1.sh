echo 'BENCHMARK START'
for i in {1..15}
do
    echo iter $i
    for j in {3,4}
    do
        for d in {5,20}
        do
            python main.py -pid $j -iid $i -d $d -pt 1 -n 'ModCMA'
            python main.py -pid $j -iid $i -d $d -pt 2 -n 'SP[20]-CMA'
            python main.py -pid $j -iid $i -d $d -pt 3 -n 'SP[10]-CMA'
        done
    done
done
echo 'BENCHMARK COMPLETE'