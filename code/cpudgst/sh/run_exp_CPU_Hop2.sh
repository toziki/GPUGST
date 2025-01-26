cd /home/lijiayu/dgst/build
cmake ..
make
#exe type path data_name T  D task_start_num(from 0) task_end_num

./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 3 4 0 500&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 3 4 501 1000&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 3 4 1001 1500&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 3 4 1501 1999&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 2 0 500&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 2 501 1000&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 2 1001 1500&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 2 1501 1999&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 3 0 500&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 3 501 1000&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 3 1001 1500&
./bin/DPBF 2 /home/lijiayu/gst/data/ orkut 5 3 1501 1999&

