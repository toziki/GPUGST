cd /home/lijiayu/gst6/gpu2dgst/build
cmake .. -D CMAKE_CXX_COMPILER=/home/lijiayu/cc/gcc-9.3.0/bin/g++ -D CMAKE_C_COMPILER=/home/lijiayu/cc/gcc-9.3.0/bin/gcc
make
#sh /home/lijiayu/gst6/gpu2dgst/sh/run_exp_GPU2_Hop.sh
#exe type path data_name T  D task_start_num(from 0) task_end_num
./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 3 4 0 40
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 4 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 5 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 3 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 4 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 5 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 3 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 4 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 5 4 0 299


# ./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 5 2 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ twitch 5 3 0 299

# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 5 2 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Github 5 3 0 299

# ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 5 2 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ musae 5 3 0 299



# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 5 2 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 5 3 0 299

#  ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 5 2 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 5 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 3 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 4 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ youtu 5 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 5 2 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 5 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 3 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 4 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ Reddit 5 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 5 2 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 5 3 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 3 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 4 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ orkut 5 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 3 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 4 4 0 299
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ dblp 5 4 0 299

rm bin/DPBF
#./bin/DPBF 2 /home/lijiayu/gst6/data/ com-amazon 3 4 0 100
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ com-amazon 4 4 0 100
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ com-amazon 5 4 0 100

# ./bin/DPBF 2 /home/lijiayu/gst6/data/ com-amazon 5 2 0 100
# ./bin/DPBF 2 /home/lijiayu/gst6/data/ com-amazon 5 3 0 100