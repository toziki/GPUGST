cd /home/lijiayu/gst/build
cmake ..  -D CMAKE_CXX_COMPILER=/home/lijiayu/cc/gcc-9.3.0/bin/g++ -D CMAKE_C_COMPILER=/home/lijiayu/cc/gcc-9.3.0/bin/gcc
make
#sh sh/run_exp_GPU1_nonHop.sh
#exe type path data_name T task_start_num(from 0) task_end_num
 ./bin/DPBF 2 /home/lijiayu/dgst/data/ twitch 3 2 20
#./bin/DPBF 2 /home/lijiayu/dgst/data/ twitch 4 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ twitch 5 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ Github 3 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ Github 4 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ Github 5 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ musae 3 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ musae 4 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ musae 5 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ com-amazon 3 0 499
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ com-amazon 4 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ com-amazon 5 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ youtu 3 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ youtu 4 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ youtu 5 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ dblp 3 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ dblp 4 0 299
#  ./bin/DPBF 2 /home/lijiayu/dgst/data/ dblp 5 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ Reddit 3 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ Reddit 4 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ Reddit 5 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ orkut 3 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ orkut 4 0 299
# ./bin/DPBF 2 /home/lijiayu/dgst/data/ orkut 5 0 299
rm bin/DPBF