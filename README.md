# Optimal Group Steiner Tree Search on GPUs

## GST_data
The data files are in the folder "data". There are six datasets: Github, Twitch, Youtube, Orkut, DBLP, Reddit. There are 8 files for each dataset. For example, the Github dataset contains the following 8 files:
1. "Github.in". This readable file contains the basic information of this dataset. The two numbers on the first line of the file represent the number of vertices and edges in the graph. The following lines have three numbers representing the two end vertices and the weight of an edge. For example, "18 14919 100" shows that there is an edge between vertex 18 and vertex 14919, with an edge weight of 100.

2. "Github_beg_pos.bin". This is a binary file. The original file has V elements, each element representing the starting position of a vertex's adjacency list. Therefore, the position of a vertex can be obtained by subtracting the starting position of the next vertex from the starting position of that vertex.

3. "Github_csr.bin". This is a binary file. The original file has E elements, and this file stores an adjacency list of vertices, where each element represents an endpoint of an edge.

4. "Github_weight.bin". This is a binary file. The original file has E elements, which store the weights of edges, with each element representing the weight of an edge.

5. "Github.g". Each line of this file represents which vertices in the graph are included in a group. For example, "g7:2705 13464 16088 16341 22323" indicates that group 7 contains five vertices: 2705, 13464, 16088, 16341, and 22323.

6. "Github3.csv". Each line of this file represents a query of size 3. For example, "2475 2384 159" indicates that the return tree of this query must contain group 2475, 2384, and 159.

7. "Github4.csv". Each line of this file represents a query of size 4. For example, "88824 76098 119691 83143" indicates that the return tree of this query must contain group 88824, 76098, 119691, and 83143.

8. "Github5.csv". Each line of this file represents a query of size 5. For example, "7 1535 901 297 1561" indicates that the return tree of this query must contain group 7, 1535, 901, 297, and 1561.
## Running code example
Here, we show how to build and run experiment on a Linux server with the Ubuntu 20.04 system, an Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz, and 1 NVIDIA GeForce RTX A6000 GPU. The environment is as follows:
- gcc version 9.3.0 (GCC)
- CUDA compiler NVIDIA 11.8.89
- cmake version 3.28.3

We will provide a detailed introduction to the experimental process as follows.

In your appropriate directory, execute the following commands:

Download the code:
```
git clone https://github.com/toziki/GPUGST.git
```
Switch the working directory to GPUGST.
```
cd GPUGST
```
Download the dataset from [OneDrive](https://1drv.ms/f/c/683d9dd9f262486b/Ek6Fl_brQzhDnI2cmhGIHxMBQ-L1ApeSqxwZKE4NBsDXSQ?e=YBWkfH). Assume that the dataset is located in the "data" folder of the working directory GPUGST.



After preparing the environment according to the above suggestions, we can use the sh files in the "sh" folder to compile and run the code.
Among them, example.sh conducts experiments on six algorithms using the Twitch dataset, with each algorithm executing 50 queries of size 3. The running instruction is:
 ```
sh sh/example.sh
 ```
The experiment results will be automatically saved as CSV files in the "data/result" folder.

Run a complete experiment of six algorithms on six datasets using run.sh. Execute 300 queries of sizes 3, 4, and 5 on each dataset. The running instruction is:

 ```
sh sh/run.sh
 ```

The other six sh files correspond to complete experiments of an algorithm on six datasets, with 300 queries of sizes 3, 4, and 5 executed on each dataset. For example, to run an experiment on GPUGST+, using instruction:

 ```
sh sh/exp_GPUGST+.sh
 ```
Taking example.sh as an example, explain the contents of the sh file as follows:
```
cd code/D-PrunedDP++
mkdir build
cd build
cmake ..
make
```
The above instructions switch to the corresponding directory of the algorithm and compile the code into an executable file.
```
./bin/cpudgst 2 ../../../ data/ Twitch 3 4 0 10
```
This instruction executes the executable file, specifying the query size, the dataset to be used and its location, the upper bound of the diameter constraint, and the start and end indices of the queries to be executed.
## GST_code
All code is located in the 'Code' folder. There are six subfolders, each corresponding to the code of one of the six experiments in section 5.1 of the paper.
- PrunedDP++. This is the PrunedDP++ version code of GST without diameter constraints.
- GPUGST. This is the GPUGST version code without diameter constraint for GST.
- GPUGST+. This is the GPUGST+ version code without diameter constraint for GST.
- D-PrunedDP++. This is the PrunedDP++ version code with diameter constraints for GST.
- D-GPUGST. This is the GPUGST version code with diameter constraints for GST.
- D-GPUGST+. This is the GPUGST+ version code with diameter constraints for GST.

In the six subfolders, there are .h, .cu, .cuh, and .cpp files used for conducting the experiments described in the paper. The .h and .cuh files are in the "include" directory, while the .cpp files are in the "src" directory. The following explanation uses code without diameter constraints as an example, and code with diameter constraints is similar.


### CPU:
- "PrunedDP++/src/main.cpp" contains the code for conducting the experiment in the paper. 
- "PrunedDP++/include/CPUNONHOP.h" contains the algorithm code for GST without diameter constraints.


### GPU
- "GPUGST/src/main.cpp" contains the code for conducting the experiment in the paper. 
- "GPUGST/include/exp_GPU_nonHop.h" contains code for reading in the graph, group, and queries.
- "GPUGST/src/DPQ.cu" contains the algorithm code.


### GPU+
- "GPUGST+/src/GSTnonHop.cu" contains the code for conducting the experiment in the paper. It also completes the tasks of reading in the graph, group, and queries.
- "GPUGST+/include/mapper_enactor.cuh" contains the overall framework of the algorithm.
- "GPUGST+/include/mapper.cuh" contains the code for performing specific operations on vertices, such as grow and merge operations.
- "GPUGST+/include/reducer.cuh" contains the code for organizing and allocating work after completing vertices operations.

