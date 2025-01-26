cd code/gpu2gst/build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num

./bin/gpu2gst 2 ../../../data/ twitch 3 0 299
./bin/gpu2gst 2 ../../../data/ twitch 4 0 299
./bin/gpu2gst 2 ../../../data/ twitch 5 0 299
./bin/gpu2gst 2 ../../../data/ Github 3 0 299
./bin/gpu2gst 2 ../../../data/ Github 4 0 299
./bin/gpu2gst 2 ../../../data/ Github 5 0 299

./bin/gpu2gst 2 ../../../data/ youtu 3 0 299
./bin/gpu2gst 2 ../../../data/ youtu 4 0 299
./bin/gpu2gst 2 ../../../data/ youtu 5 0 299
./bin/gpu2gst 2 ../../../data/ dblp 3 0 299
./bin/gpu2gst 2 ../../../data/ dblp 4 0 299
./bin/gpu2gst 2 ../../../data/ dblp 5 0 299
./bin/gpu2gst 2 ../../../data/ Reddit 3 0 299
./bin/gpu2gst 2 ../../../data/ Reddit 4 0 299
./bin/gpu2gst 2 ../../../data/ Reddit 5 0 299
./bin/gpu2gst 2 ../../../data/ orkut 3 0 299
./bin/gpu2gst 2 ../../../data/ orkut 4 0 299
./bin/gpu2gst 2 ../../../data/ orkut 5 0 299
rm bin/gpu2gst
