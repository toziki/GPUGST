cd code/cpudgst
mkdir build
cd build
cmake ..
make 
./bin/cpudgst 2 ../../../data/ twitch 3 4 0 10

# cd ../../../code/cpugst
# mkdir build
# cd build
# cmake ..
# make 
#  ./bin/cpugst 1 ../../../data/ twitch 3 0 10

cd ../../../code/gpu1gst
mkdir build
cd build
cmake ..
make 
 ./bin/gpu1gst 2 ../../../data/ twitch 3 0 10

 cd ../../../code/gpu1dgst
mkdir build
cd build
cmake ..
make 
 ./bin/gpu1dgst 2 ../../../data/ twitch 3 4 0 10

#  cd ../../../code/gpu2gst
# mkdir build
# cd build
# cmake ..
# make 
#  ./bin/gpu2gst 1 ../../../data/ twitch 3 0 10

#  cd ../../../code/gpu2dgst
# mkdir build
# cd build
# cmake ..
# make 
#  ./bin/gpu2dgst 1 ../../../data/ twitch 3 4 0 10