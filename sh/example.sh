cd cpudgst
mkdir build
cd build
cmake ..
make 
./bin/cpudgst 2 /home/lijiayu/gst6/data/ twitch 3 4 0 49

cd ../../cpugst
mkdir build
cd build
cmake ..
make 
 ./bin/cpugst 1 /home/lijiayu/gst6/data/ twitch 3 0 49

cd ../../gpu1gst
mkdir build
cd build
cmake ..
make 
 ./bin/gpu1gst 1 /home/lijiayu/gst6/data/ twitch 3 0 49

 cd ../../gpu1dgst
mkdir build
cd build
cmake ..
make 
 ./bin/gpu1dgst 1 /home/lijiayu/gst6/data/ twitch 3 4 0 49

 cd ../../gpu2gst
mkdir build
cd build
cmake ..
make 
 ./bin/gpu2gst 1 /home/lijiayu/gst6/data/ twitch 3 0 49

 cd ../../gpu2dgst
mkdir build
cd build
cmake ..
make 
 ./bin/gpu2dgst 1 /home/lijiayu/gst6/data/ twitch 3 4 0 49