cd code/gpu1gst/build
cmake .. 
make
#sh sh/run_exp_GPU1_nonHop.sh
#exe type path data_name T task_start_num(from 0) task_end_num
 ./bin/DPBF 2 ../../../data/ twitch 3 2 20
#./bin/DPBF 2 ../../../data/ twitch 4 0 299
# ./bin/DPBF 2 ../../../data/ twitch 5 0 299
# ./bin/DPBF 2 ../../../data/ Github 3 0 299
# ./bin/DPBF 2 ../../../data/ Github 4 0 299
# ./bin/DPBF 2 ../../../data/ Github 5 0 299
# ./bin/DPBF 2 ../../../data/ musae 3 0 299
# ./bin/DPBF 2 ../../../data/ musae 4 0 299
# ./bin/DPBF 2 ../../../data/ musae 5 0 299
# ./bin/DPBF 2 ../../../data/ com-amazon 3 0 499
# ./bin/DPBF 2 ../../../data/ com-amazon 4 0 299
# ./bin/DPBF 2 ../../../data/ com-amazon 5 0 299
# ./bin/DPBF 2 ../../../data/ youtu 3 0 299
# ./bin/DPBF 2 ../../../data/ youtu 4 0 299
# ./bin/DPBF 2 ../../../data/ youtu 5 0 299
# ./bin/DPBF 2 ../../../data/ dblp 3 0 299
# ./bin/DPBF 2 ../../../data/ dblp 4 0 299
#  ./bin/DPBF 2 ../../../data/ dblp 5 0 299
# ./bin/DPBF 2 ../../../data/ Reddit 3 0 299
# ./bin/DPBF 2 ../../../data/ Reddit 4 0 299
# ./bin/DPBF 2 ../../../data/ Reddit 5 0 299
# ./bin/DPBF 2 ../../../data/ orkut 3 0 299
# ./bin/DPBF 2 ../../../data/ orkut 4 0 299
# ./bin/DPBF 2 ../../../data/ orkut 5 0 299
rm bin/DPBF