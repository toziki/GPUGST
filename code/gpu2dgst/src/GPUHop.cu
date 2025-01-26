#include "header.h"
#include "util.h"
#include "mapper.cuh"
#include "reducer.cuh"
#include "wtime.h"
#include "barrier.cuh"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include "mapper_enactor.cuh"
#include <thrust/detail/minmax.h>
#include <thrust/fill.h>
#define THREAD_PER_BLOCK 512
using namespace std;
/*user defined vertex behavior function*/

__inline__ __host__ __device__ feature_t user_mapper_push // reduce edge_compute_push
	(vertex_t src,
	 vertex_t dest,
	 feature_t level,
	 index_t *beg_pos,
	 weight_t edge_weight,
	 feature_t *vert_status,
	 feature_t *vert_status_prev)
{
	// if (vert_status_prev[src] == vert_status[src])
	return vert_status[src] + edge_weight;
	// else
	//     return INFTY;
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_push(vertex_t vert_id,
														 feature_t level,
														 vertex_t *adj_list,
														 index_t *beg_pos,
														 feature_t *vert_status,
														 feature_t *vert_status_prev)
{
	return (vert_status[vert_id] != vert_status_prev[vert_id]);
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ feature_t user_mapper_pull(vertex_t src,
														  vertex_t dest,
														  feature_t level,
														  index_t *beg_pos,
														  weight_t edge_weight,
														  feature_t *vert_status,
														  feature_t *vert_status_prev)
{
	// return vert_status[src] + edge_weight;
	return vert_status[src] + edge_weight;
}

/*user defined vertex behavior function*/
__inline__ __host__ __device__ bool vertex_selector_pull // reduce
	(vertex_t vert_id,
	 feature_t level,
	 vertex_t *adj_list,
	 index_t *beg_pos,
	 feature_t *vert_status,
	 feature_t *vert_status_prev)
{
	return (beg_pos[vert_id] != beg_pos[vert_id + 1]);
	// return (vert_status[vert_id] != vert_status_prev[vert_id]);
	// return (vert_status[vert_id] != vert_status_prev[vert_id] || vert_status[vert_id] == INFTY);
	// return true;
}

__device__ cb_reducer vert_selector_push_d = vertex_selector_push;
__device__ cb_reducer vert_selector_pull_d = vertex_selector_pull;
__device__ cb_mapper vert_behave_push_d = user_mapper_push;
__device__ cb_mapper vert_behave_pull_d = user_mapper_pull;
/*init GST*/
__global__ void count_used(int *tree,int val1,int val2, int width, int *counts,int D,int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		int vline = idx*val1;
		for (int x = 1; x < width; x++)
		{
			for (size_t j = 0; j <= D; j++)
			{
				if(tree[vline+x*val2+j]!=inf)
				atomicAdd(counts,1);
			}
			
			
			
		}
	}
}

int main(int args, char **argv)
{
	string path = argv[2], data_name = argv[3];
	int T = atoi(argv[4]), D = atoi(argv[5]), task_start_num = atoi(argv[6]), task_end_num = atoi(argv[7]);
	// double tm_map, tm_red, tm_scan;   无直径约束gpu2
	string file_beg = path + data_name + "_beg_pos.bin";
	string file_adj = path + data_name + "_csr.bin";
	string file_weight = path + data_name + "_weight.bin";

	const char *file_beg_pos = file_beg.c_str();
	const char *file_adj_list = file_adj.c_str();
	const char *file_weight_list = file_weight.c_str();

	int blk_size = 512;
	
	int *non_overlapped_group_gpu, *non_overlapped_group_pointer, *can_find;
	// Read graph to CPU
	graph<long, long, long, vertex_t, index_t, weight_t>
		*ginst = new graph<long, long, long, vertex_t, index_t, weight_t>(file_beg_pos, file_adj_list, file_weight_list);
	ginst->read_gi(path + data_name + ".g", path + data_name + to_string(T) + ".csv");
	int G = ginst->inquire[0].size(), V = ginst->vert_count, solve = 0;
	int width = 1 << G;
	int problem_size = V * width * (D + 1), sum_cost = 0;
	non_overlapped_group_sets s = graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(width - 1);
	cudaMallocManaged((void **)&non_overlapped_group_gpu, sizeof(int) * s.length);
	cudaMallocManaged((void **)&non_overlapped_group_pointer, sizeof(int) * (width + 1));
	cudaMemcpy(non_overlapped_group_gpu, s.non_overlapped_group_sets_IDs.data(), sizeof(int) * s.length, cudaMemcpyHostToDevice);
	cudaMemcpy(non_overlapped_group_pointer, s.non_overlapped_group_sets_IDs_pointer_host.data(), (width + 1) * sizeof(int), cudaMemcpyHostToDevice);
	meta_data mdata(ginst->vert_count, ginst->edge_count, width, D);
	std::vector<int> contain_group_vertices;
	cout << "width " << width << " V " << V << endl;
	int VAL1 = (1 << G) * (D + 1), VAL2 = (D + 1);
	int *host_tree = new int[problem_size];
	feature_t *level, *level_h;
	cudaMalloc((void **)&level, 10 * sizeof(feature_t));
	cudaMallocHost((void **)&level_h, 10 * sizeof(feature_t));
	cudaMemset(level, 0, sizeof(feature_t));
	cb_reducer vert_selector_push_h;
	cb_reducer vert_selector_pull_h;
	cudaMemcpyFromSymbol(&vert_selector_push_h, vert_selector_push_d, sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_selector_pull_h, vert_selector_pull_d, sizeof(cb_reducer));

	cb_mapper vert_behave_push_h;
	cb_mapper vert_behave_pull_h;
	double sum_time = 0;
	cudaMemcpyFromSymbol(&vert_behave_push_h, vert_behave_push_d, sizeof(cb_reducer));
	cudaMemcpyFromSymbol(&vert_behave_pull_h, vert_behave_pull_d, sizeof(cb_reducer));
	// Init three data structures
	gpu_graph ggraph(ginst);

	ggraph.merge_groups_d = non_overlapped_group_gpu, ggraph.merge_pointer_d = non_overlapped_group_pointer;
	cout << "width " << mdata.width << " V " << ginst->vert_count << endl;
	Barrier global_barrier(BLKS_NUM);
	mapper compute_mapper(ggraph, mdata, vert_behave_push_h, vert_behave_pull_h, width);
	reducer worklist_gather(ggraph, mdata, vert_selector_push_h, vert_selector_pull_h, width);
	H_ERR(cudaDeviceSynchronize());
	ofstream outputFile;
	outputFile.precision(8);
	outputFile.setf(ios::fixed);
	outputFile.setf(ios::showpoint);
	outputFile.open(path+"result/exp_GPU2_Hop_" + data_name + "_T" + to_string(T) + "_" + "D" + to_string(D) + "_" + to_string(task_start_num) + "-" + to_string(task_end_num) + ".csv");

	outputFile << "task_ID,task,GPU2_Hop_time,GPU2_Hop_cost,GPU2_Hop_memory,counts,process_num" << endl;

	for (int i = 0; i < ginst->inquire.size(); i++)
	{

		for (size_t j = 0; j < ginst->inquire[i].size(); j++)
		{
			// ginst->inquire[i][j] -= (V);
			// cout << ginst->inquire[i][j] << " ";
		}
		// cout << endl;
	}

	for (int ii = task_start_num; ii <= task_end_num; ii++)
	{
		std::cout << "------------------------------------------------------------" << endl;
		cout << data_name << " interation " << ii << " "<<endl;
		string task = "";
		for (size_t j = 0; j < ginst->inquire[ii].size(); j++)
		{
			task += to_string(ginst->inquire[ii][j]) + " ";
		}
		//* necessary for high diameter graph, e.g., euro.osm and roadnet.ca
		// mapper_merge_push(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
		cudaMemset(level, 0, 10 * sizeof(feature_t));
		std::fill(host_tree, host_tree + problem_size, inf);
		cudaMemcpy(mdata.vert_status_prev, host_tree, problem_size * sizeof(int), cudaMemcpyHostToDevice);
		set_max_ID(ginst->group_graph, ginst->inquire[ii], host_tree, contain_group_vertices, width, D); // 在置初值前传递参数到prev
		for (auto it = contain_group_vertices.begin(); it != contain_group_vertices.end(); it++)
		{
			int v = *it;
			int group_set_ID_v = get_max(v, host_tree, width, VAL1, VAL2); /*time complexity: O(|Gamma|)*/
			for (size_t i = 1; i <= group_set_ID_v; i++)
			{
				if ((i | group_set_ID_v) == group_set_ID_v)
				{
					host_tree[v * VAL1 + i * VAL2] = 0;
				}
			}
		}
		//  for (int i = 0; i < V; i++)
		// {
		//     for (int j = 0; j < width; j++)
		//     {
		//         for (int d = 0; d <= D; d++)
		//         {
		// 				if(host_tree[i*VAL1+j*VAL2+d]!=inf)
		//                 printf("i=%d p=%d d=%d c=%d\n",i,j,d,host_tree[i*VAL1+j*VAL2+d]);
		//         }
		//     }
		// }

		cudaMemcpy(mdata.vert_status, host_tree, problem_size * sizeof(int), cudaMemcpyHostToDevice);
		// mapper_hybrid_push_merge(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier, 0);

		double time = wtime();
		// mapper_hybrid_push_merge(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier, 0);

		balanced_push(blk_size, level, ggraph, mdata, compute_mapper, worklist_gather, global_barrier);
		double ftime = wtime() - time; // 一阶段推的耗时
		cudaMemcpy(level_h, level, 10 * sizeof(feature_t), cudaMemcpyDeviceToHost);
		//std::cout << "first round iteration: " << level_h[0] << " queue size " << level_h[1] << " future work " << level_h[2] << " overflow " << level_h[3] << "\n";
		cudaMemcpy(host_tree, mdata.vert_status, problem_size * sizeof(int), cudaMemcpyDeviceToHost);
		// for (size_t i = 0; i < V; i++)
		// {
		// 	cout << i << "  ";
		// 	for (size_t j = 1; j < width; j++)
		// 	{
		// 		for (size_t d1 = 0; d1 <= D; d1++)
		// 		{
		// 			cout << "i= "<<i<<" j= "<<j<<" d="<<d1<<" cost "<<host_tree[i * VAL1 + j*VAL2+d1] << " ";
		// 		}
		// 		cout << endl;

		// 	}
		// 	cout << endl;
		// }
		int minc = inf;
		for (size_t i = 0; i < V; i++)
		{
			for (size_t j = 0; j < D; j++)
			{
				minc = min(minc, host_tree[i * VAL1 + (width - 1) * VAL2 + j]);
			}
		}
		cout << "min cost " << minc << endl;
		if (minc != inf)
		{
			sum_cost += minc;
			sum_time += ftime;
			solve++;
		}

		time = wtime() - time;
		int *set_num;
		cudaMallocManaged((void **)&set_num,sizeof(int));
		count_used<<<(V + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(mdata.vert_status,VAL1,VAL2, width, set_num,D, V);
		cudaDeviceSynchronize();
		//cout<<"set num "<<*set_num<<" queue size "<<level_h[5]<<" V*width*D "<<V*width*D<<endl;
		int mem_MB = (level_h[5]+V*width*D+*set_num);
		outputFile << ii << "," << task << "," << ftime << "," << minc << "," <<mem_MB <<","<<*set_num<<","<<level_h[6]<< endl;

		//std::cout << "Staged time: " << ftime << "\n"; // 三段时间
		cudaMemcpy(level_h, level, sizeof(feature_t), cudaMemcpyDeviceToHost);
	}
	delete[] host_tree;

	mdata.release();
	ggraph.release();
	// delete[] gpu_dist;
	return 0;
}
