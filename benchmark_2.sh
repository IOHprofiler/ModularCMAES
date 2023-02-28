# benchmarking initialization techniques
echo 'BENCHMARK START'
for i in {1,3,5}
do
    echo iter $i
    for j in {1..3}
    do
        echo iter iter $j
        python main.py -pid 3 -iid $i -d 5 -pt 2 -n 'SP[20]-CMA'
        python main.py -pid 3 -iid $i -d 20 -pt 2 -n 'SP[20]-CMA'
        python main.py -pid 4 -iid $i -d 5 -pt 2 -n 'SP[20]-CMA'
        python main.py -pid 4 -iid $i -d 20 -pt 2 -n 'SP[20]-CMA'
        # python main.py -pid 3 -iid $i -d 5 -pt 3 -n 'SP[10]-CMA-UNI'
        # python main.py -pid 3 -iid $i -d 20 -pt 3 -n 'SP[10]-CMA-UNI'
        # python main.py -pid 4 -iid $i -d 5 -pt 3 -n 'SP[10]-CMA-UNI'
        # python main.py -pid 4 -iid $i -d 20 -pt 3 -n 'SP[10]-CMA-UNI'


        python main.py -pid 3 -iid $i -d 5 -pt 2 -n 'SP[20]-CMA-LHS' -s lhs
        python main.py -pid 3 -iid $i -d 20 -pt 2 -n 'SP[20]-CMA-LHS' -s lhs
        python main.py -pid 4 -iid $i -d 5 -pt 2 -n 'SP[20]-CMA-LHS' -s lhs
        python main.py -pid 4 -iid $i -d 20 -pt 2 -n 'SP[20]-CMA-LHS' -s lhs
        # python main.py -pid 3 -iid $i -d 5 -pt 3 -n 'SP[10]-CMA-LHS' -s lhs
        # python main.py -pid 3 -iid $i -d 20 -pt 3 -n 'SP[10]-CMA-LHS' -s lhs
        # python main.py -pid 4 -iid $i -d 5 -pt 3 -n 'SP[10]-CMA-LHS' -s lhs
        # python main.py -pid 4 -iid $i -d 20 -pt 3 -n 'SP[10]-CMA-LHS' -s lhs


        python main.py -pid 3 -iid $i -d 5 -pt 2 -n 'SP[20]-CMA-SOBOL' -s sobol
        python main.py -pid 3 -iid $i -d 20 -pt 2 -n 'SP[20]-CMA-SOBOL' -s sobol
        python main.py -pid 4 -iid $i -d 5 -pt 2 -n 'SP[20]-CMA-SOBOL' -s sobol
        python main.py -pid 4 -iid $i -d 20 -pt 2 -n 'SP[20]-CMA-SOBOL' -s sobol
        # python main.py -pid 3 -iid $i -d 5 -pt 3 -n 'SP[10]-CMA-SOBOL' -s sobol
        # python main.py -pid 3 -iid $i -d 20 -pt 3 -n 'SP[10]-CMA-SOBOL' -s sobol
        # python main.py -pid 4 -iid $i -d 5 -pt 3 -n 'SP[10]-CMA-SOBOL' -s sobol
        # python main.py -pid 4 -iid $i -d 20 -pt 3 -n 'SP[10]-CMA-SOBOL' -s sobol
    done    
done
echo 'BENCHMARK COMPLETE'