# Optimal Group Steiner Tree Search on GPUs

## GST_data
The data files are zipped together, and should be unzipped first. There are six datasets: Github, Twitch, Youtube, Orkut, DBLP, Reddit.There are 8 files for each dataset. For example, the Github dataset contains the following 8 files. 
1. "Github.in". This readable file contains the basic information of this dataset.The two numbers on the first line of the file represent the number of points and edges in the graph. The following lines have three numbers representing the two endpoints and weights of an edge. For example, "18 14919 100" shows that there is a edge between vertex 18 and vertex 14919, with the edge weight of 100.

2. "Github_beg_pos.bin".Binary file. The original file has V elements, each element representing the starting position of a point's adjacency list. Therefore, the position of a point can be obtained by subtracting the starting position of the next point from the starting position of that point.

3. "Github_csr.bin".Binary file. The original file has E elements, and this file stores an adjacency list of points, where each element represents an endpoint of an edge.

4. "Github_weight.bin".Binary files. The original file has E elements, which store the weights of edges, with each element representing the weight of an edge.

5. "Github.g". Each line of this file represents which vertexs in the graph are included in a group. For example, "g7:2705 13464 16088 16341 22323" indicates that group 7 contains five vertexs: 2705,13464, 16088, 16341, and 22323

6. "Github3.csv". Each line of this file represents a query of size 3, for example, "2475 2384 159" indicates that the return tree of this query must contain  group 2475, 2384 and 159.

7. "Github4.csv". Each line of this file represents a query of size 4, for example, "88824 76098 119691 83143" indicates that the return tree of this query must contain  group 88824, 76098, 119691 and 83143.

8. "Github5.csv". Each line of this file represents a query of size 5, for example, "7 1535 901 297 1561" indicates that the return tree of this query must contain  group 7, 1535, 901, 297, and 1561.
## Build & Run
Here, we show how to build & run experiment on a Linux server with the Ubuntu 20.04 system, Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz, and 1 NVIDIA GeForce RTX A6000 GPUs. The environment is as follows.
- gcc version 9.3.0 (GCC)
- CUDA compiler NVIDIA 11.8.89
- cmake version 3.28.3

You can compile and run the code by executing the. sh file with the sh command. The sh file is stored in the/sh folder. Among them, example. sh conducted experiments on six algorithms on the Twitch dataset, with each algorithm executing 50 queries of size 3. Run a complete experiment of six algorithms on six datasets using run.sh. Execute 300 queries of sizes 3, 4, and 5 on each dataset. The other six sh files correspond to complete experiments of an algorithm on six datasets, with 300 queries of sizes 3, 4, and 5 executed on each dataset.
Please modify the folder directory corresponding to the data in the sh file.
In the GST folder, the sh file can be executed using the following command:
cd sh
sh example.sh


## GST_code
All code is located in the code folder.There are six folders correspond to the codes of six experiments.
- cpugst. This is the CPU version code of GST without diameter constraints.
- gpu1gst. This is the GPU4GST version code without diameter constraint for GST.
- gpu2gst. This is the GPU4GST+ version code without diameter constraint for GST.
- cpudgst. This is the CPU version code with diameter constraints for GST.
- gpu1dgst. This is the GPU4GST version code with diameter constraints for GST.
- gpu2dgst. This is the GPU4GST+ version code with diameter constraints for GST.

In six folders,there are some h,cu,duh,and cpp files for conducting experiments in the paper. The h and cuh files are at "include", while the cpp files are at "src".The following explanation uses code without diameter constraints as an example, and code with diameter constraints is similar to it.


### CPU:
- "cpugst/src/main.cpp" contains codes for conducting the experiment in the paper. 
- "cpugst/include/CPUNONHOP.h" contains the algorithm code for GST without diameter constraints.


### GPU
- "gpu1gst/src/main.cpp" contains codes for conducting the experiment in the paper. 
- "gpu1gst/include/exp_GPU_nonHop.h" contains code for reading in graph, group, and queries.
- "gpu1gst/src/DPQ.cu" file contains the algorithm code for GPU4GST.


### GPU+
- "gpu2gst/src/GSTnonHop.cu" contains codes for conducting the experiment in the paper.It also completes the tasks of reading in graph, group, and queries.
- "gpu2gst/include/mapper_enactor.cuh" contains the overall framework of the GPU4GST+ algorithm.
- "gpu2gst/include/mapper.cuh" file contains the code for GPU4GST+ to perform specific operations on vertexs, such as grow and merge operations.
- "gpu2gst/include/reducer.cuh" file  contains the code for organizing and allocating work after GPU4GST+ completes vertexs operations.

