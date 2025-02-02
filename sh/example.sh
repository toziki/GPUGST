#sh sh/example.sh
cd code/D-PrunedDP++
mkdir build
cd build
cmake ..
make 
./bin/D-PrunedDP++ 2 ../../../data/ Twitch 3 4 0 10

cd ../../../code/PrunedDP++
mkdir build
cd build
cmake ..
make 
 ./bin/PrunedDP++ 1 ../../../data/ Twitch 3 0 10

cd ../../../code/GPUGST
mkdir build
cd build
cmake ..
make 
 ./bin/GPUGST 2 ../../../data/ Twitch 3 0 10

 cd ../../../code/D-GPUGST
mkdir build
cd build
cmake ..
make 
 ./bin/D-GPUGST 2 ../../../data/ Twitch 3 4 0 10

 cd ../../../code/GPUGST+
mkdir build
cd build
cmake ..
make 
 ./bin/GPUGST+ 1 ../../../data/ Twitch 3 0 10

 cd ../../../code/D-GPUGST+
mkdir build
cd build
cmake ..
make 
 ./bin/D-GPUGST+ 1 ../../../data/ Twitch 3 4 0 10