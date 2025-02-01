cd /home/lijiayu/gst6/gpu2gst/build
cmake .. -D CMAKE_CXX_COMPILER=/home/lijiayu/cc/gcc-9.3.0/bin/g++ -D CMAKE_C_COMPILER=/home/lijiayu/cc/gcc-9.3.0/bin/gcc
make
#sh /home/lijiayu/gst6/gpu2gst/sh/run_exp_GPU2_nonHop.sh
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 3 0 100
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 5 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 5 0 299
#  ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 5 0 299

# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 5 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 5 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 5 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 5 0 299
rm bin/DPBF
