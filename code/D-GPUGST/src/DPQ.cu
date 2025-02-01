#include <chrono>
#include <DPQ.cuh>
#include <thrust/device_vector.h>
using namespace std;



struct queue_element_d
{
	int v, p, d;

    queue_element_d(int _v = 0, int _p = 0, int _d = 0)
        : v(_v), p(_p), d(_d) {}
};

void set_max_ID(graph_v_of_v_idealID &group_graph, std::vector<int> &cumpulsory_group_vertices, node *host_tree, std::unordered_set<int> &contain_group_vertices,int val1,int val2)
{
	int bit_num = 1, v;
	for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++, bit_num <<= 1)
	{
		for (size_t to = 0; to < group_graph[*it].size(); to++)
		{
			v = group_graph[*it][to].first;
			host_tree[v*val1+ bit_num * val2].cost = 0;
			contain_group_vertices.insert(v);
		}
	}
}
int get_max(int vertex, node *host_tree,int width, int val1,int val2)
{
	int re = 0;
	for (size_t i = 1; i < width; i <<= 1)
	{
		if (host_tree[vertex*val1+ i * val2].cost == 0)
		{
			re += i;
		}
	}

	return re;
}
inline int graph_v_of_v_idealID_DPBF_vertex_group_set_ID_gpu(int vertex, graph_v_of_v_idealID &group_graph,
													  std::unordered_set<int> &cumpulsory_group_vertices)
{

	/*time complexity: O(|Gamma|); this function returns the maximum group set ID for a single vertex*/
	// if group i have edge to v,v will give bit i value 1;
	int ID = 0;
	int pow_num = 0;
	for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++)
	{
		if (graph_v_of_v_idealID_contain_edge(group_graph, vertex, *it))
		{ // vertex is in group *it
			ID = ID + (1 << pow_num);
		}
		pow_num++;
	}

	return ID;
}

__global__ void Relax(
    int *pointer, int *edge, int *edge_weight,
    queue_element_d *queue, int *queue_size,
    queue_element_d *queue2, int *queue_size2,
    node *host_tree, int *updated, 
    int VAL1, int VAL2, int group_sets_ID_range, int D) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *queue_size) {
        queue_element_d now = queue[idx];
        int v = now.v;
        int p = now.p;
        int d = now.d;
        if (p == group_sets_ID_range) {
            return;
        }
        //grow
        int cost = host_tree[v * VAL1 + p * VAL2 + d].cost;
        if (d < D) {
            for (int i = pointer[v]; i < pointer[v + 1]; i++) {
                int u = edge[i];
                int z = edge_weight[i];
                int id = u * VAL1 + p * VAL2 + (d + 1);
                int old = atomicMin(&host_tree[id].cost, cost + z);
                if (old > cost + z) {
                    host_tree[id].type = 1;
                    host_tree[id].u = v;
                    int t = atomicCAS(&updated[id], 0, 1);
                    if (!t) {
                        int now = atomicAdd(queue_size2, 1);
                        queue2[now].v = u;
                        queue2[now].p = p;
                        queue2[now].d = d + 1;
                    }
                }
            }
        }
        //merge
        int p1 = p, d1 = d;
        int mask = group_sets_ID_range ^ p;
        for (int p2 = mask; p2 > 0; p2 = (p2 - 1) & mask) {
            for (int d2 = 0; d2 <= D - d1; d2++) {
                int p1_cup_p2 = p1 | p2;
                int new_d = max(d1, d2);
                int merge_tree_cost = cost + host_tree[v * VAL1 + p2 * VAL2 + d2].cost;
                int id = v * VAL1 + p1_cup_p2 * VAL2 + new_d;
                int old = atomicMin(&host_tree[id].cost, merge_tree_cost);
                if (old > merge_tree_cost) {
                    int t = atomicCAS(&updated[id], 0, 1);
                    if (!t) {
                        int now = atomicAdd(queue_size2, 1);
                        queue2[now].v = v;
                        queue2[now].p = p1_cup_p2;
                        queue2[now].d = new_d;
                    }
                    host_tree[id].type = 2;
                    host_tree[id].p1 = p1;
                    host_tree[id].p2 = p2;
                    host_tree[id].d1 = d1;
                    host_tree[id].d2 = d2;
                }
            }
        }
    }
}

__global__ void count_set(node *tree,int val1,int val2,int width,int inf, int N,int D,int *counts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		int vline = idx * val1;
		for (int x = 1; x < width; x++)
		{
            for (size_t j = 0; j <= D; j++)
            {
          	if(tree[vline+x*val2+j].cost!=inf)
			{
			atomicAdd(counts,1);
			}		
            }
            
			
		}
	}
}
graph_hash_of_mixed_weighted DPBF_gpu(CSR_graph &graph, std::vector<int> &cumpulsory_group_vertices, graph_v_of_v_idealID &group_graph, graph_v_of_v_idealID &input_graph, int D,double *rt,int &real_cost,int &RAM,records &ret)

{
    int N = input_graph.size();
    int G = cumpulsory_group_vertices.size();
    int group_sets_ID_range = (1 << G) - 1;
        int width  = 1<<G;
    int V = N * (1 << G) * (D + 3);
    int VAL1 = (1 << G) * (D + 1);
    int VAL2 = (D + 1);
    
    node *host_tree; // node host_tree[N][1 << G][D + 3];
    queue_element_d *queue, *queue2;
    int *queue_size, *queue_size2, *updated,max_queue_size=0;

    int *edge = graph.all_edge;
    int *edge_weight = graph.all_edge_weight;
    int *pointer = graph.all_pointer;

    cudaMallocManaged((void **)&host_tree, sizeof(node) * V);
    cudaMallocManaged((void**)&queue, V * sizeof(queue_element_d));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&queue2, V * sizeof(queue_element_d));
    cudaMallocManaged((void**)&queue_size2, sizeof(int));
    cudaMallocManaged((void**)&updated, sizeof(int) * V);

    for (int v = 0; v < N; v++) {
        for (int p = 0; p <= group_sets_ID_range; p++) {
            for (int d = 0; d <= D; d++) {
                host_tree[v * VAL1 + p * VAL2 + d].cost = inf;
            }
        }
    }
    *queue_size = 0;
    *queue_size2 = 0;
    std::unordered_set<int> contain_group_vertices;
	set_max_ID(group_graph, cumpulsory_group_vertices, host_tree, contain_group_vertices,VAL1,VAL2);
    for(int v = 0; v < N; v++){
        int group_set_ID_v =get_max(v, host_tree, width,VAL1,VAL2);
        for(int i = 1; i <= group_set_ID_v; i <<= 1){
            if(i & group_set_ID_v){
                int id = v * VAL1 + i * VAL2;
                host_tree[id].cost = 0;
                host_tree[id].type = 0;
                queue[*queue_size] = queue_element_d(v, i, 0);
                (*queue_size)++;
            }
        }
    }
    
    int threadsPerBlock = 1024;
    int numBlocks = 0;
    
    auto pbegin = std::chrono::high_resolution_clock::now();
    int tot_process = 0;
    while (*queue_size > 0) {
        cudaMemset(updated, 0, V * sizeof(int));
        numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
        tot_process+=*queue_size;
        Relax <<< numBlocks, threadsPerBlock >>> (
            pointer, edge, edge_weight,
            queue, queue_size,
            queue2, queue_size2,
            host_tree, updated, 
            VAL1, VAL2, group_sets_ID_range, D
                                                    );
        cudaDeviceSynchronize();

        //the function get the prefix sum of updated
        //?
        swap(queue, queue2);
        swap(queue_size, queue_size2);
        max_queue_size = max(max_queue_size,*queue_size);
        *queue_size2 = 0;
    }
    
  auto pend = std::chrono::high_resolution_clock::now();
	double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(pend - pbegin).count() / 1e9; // s
	*rt = runningtime;
//	std ::cout << "gpu cost time " << runningtime << std ::endl;
    std :: queue<queue_element_d> Q;
    graph_hash_of_mixed_weighted solution_tree;
    int ans = inf;
    queue_element_d pos;
    for (int i = 0; i < N; i++) {
        for (int d = 0; d <= D; d++) {
            int now = host_tree[VAL1 * i + VAL2 * group_sets_ID_range + d].cost;
            if (ans > now) {
                ans = now;
                pos = queue_element_d(i, group_sets_ID_range, d);
            }
        }
    }
       for (int i = 0; i < N; i++) {
        for (int d = 0; d <= D; d++) {
            int now = host_tree[VAL1 * i + VAL2 * group_sets_ID_range + d].cost;
            if (ans == now) {
              //  cout<<"d: "<<d<<endl;
            }
        }
    }
   
    real_cost = ans;
  
    Q.push(pos);
 
   	int *counts;
	cudaMallocManaged((void **)&counts,sizeof(int));
	count_set<<<(V + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(host_tree,VAL1,VAL2,width,inf,N,D,counts);
    cudaDeviceSynchronize();
    ret.counts = *counts;
    ret.process_queue_num = tot_process;
	RAM = (*counts+max_queue_size+N*width*D);
    cudaFree(host_tree);
    cudaFree(queue);
    cudaFree(queue2);
    cudaFree(updated);  
    cudaFree(queue_size);
    cudaFree(queue_size2);
    return solution_tree;
}
