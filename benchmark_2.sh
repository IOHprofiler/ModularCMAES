echo 'BENCHMARK START'
for i in {1..3}
do
    echo iter $i
    for j in {3,4}
    do
        for d in {5,20}
        do
            python main.py -pid $j -iid $i -d $d -pt 1 -n 'ModCMA'
            python main.py -pid $j -iid $i -d $d -pt 2 -n 'SP[50]-CMA-SVM' -ic svm
            python main.py -pid $j -iid $i -d $d -pt 3 -n 'SP[20]-CMA-SVM' -ic svm
            python main.py -pid $j -iid $i -d $d -pt 4 -n 'SP[10]-CMA-SVM' -ic svm
        done
    done
done
echo 'BENCHMARK COMPLETE'