#sh sh/example.sh
cd code/D-PrunedDP++
mkdir build
cd build
cmake ..
make 
./bin/cpudgst 2 ../../../data/ Twitch 3 4 0 10

cd ../../../code/PrunedDP++
mkdir build
cd build
cmake ..
make 
 ./bin/cpugst 1 ../../../data/ Twitch 3 0 10

cd ../../../code/GPUGST
mkdir build
cd build
cmake ..
make 
 ./bin/gpu1gst 2 ../../../data/ Twitch 3 0 10

 cd ../../../code/D-GPUGST
mkdir build
cd build
cmake ..
make 
 ./bin/gpu1dgst 2 ../../../data/ Twitch 3 4 0 10

 cd ../../../code/GPUGST+
mkdir build
cd build
cmake ..
make 
 ./bin/gpu2gst 1 ../../../data/ Twitch 3 0 10

 cd ../../../code/D-GPUGST+
mkdir build
cd build
cmake ..
make 
 ./bin/gpu2dgst 1 ../../../data/ Twitch 3 4 0 10