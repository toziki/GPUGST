cd /home/lijiayu/gst/build 
cmake ..
make
#exe       type path             data_name T  D task_start_num(from 0) task_end_num
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 4 678 1000&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 4 1001 1500&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 4 1501 1999&

./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 5 158 300&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 5 301 600&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 5 601 900&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 5 901 1100&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 5 1101 1400&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 5 1401 1700&
./bin/DPBF 1 /home/lijiayu/gst/data/ orkut 5 1701 1999&

