p="10000000 64 3 1 1000 0"
alias c="g++ -Wall -std=c++17 -g -O3 -fopenmp -D_GLIBCXX_PARALLEL bdm-performance-test-bench.cc -o bdm-performance-test-bench"
alias r='echo "Parameter: $p" && taskset -c 0-13,28-41 ./bdm-performance-test-bench $p'
alias cr='c && r'
alias t1="export OMP_NUM_THREADS=1"
alias tall="unset OMP_NUM_THREADS"
alias vtune='taskset -c 0-13,28-41 amplxe-cl -collect advanced-hotspots -- ./bdm-performance-test-bench $p'
