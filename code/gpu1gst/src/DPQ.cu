
#include <DPQ.cuh>
#include <thrust/copy.h>
#include<thrust/detail/minmax.h>
#include<chrono>
using namespace std;

typedef struct queue_element
{
	int v, p;
} queue_element;

__device__ __forceinline__ int get_lb(node *tree, int vline, int x_slash)
{
	// 计算lower bound需要的参数 ： 状态status 要计算的v,p
	int ret = 0;
	for (int i = 1; i <= x_slash; i <<= 1)
	{
		if ((x_slash & i) && ret < tree[vline + i].c)
		{
			ret = tree[vline + i].c;
		}
	}
	return ret;
}
std::vector<int> non_overlapped_group_sets_IDs_pointer_host, non_overlapped_group_sets_IDs;
void set_max_ID(graph_v_of_v_idealID &group_graph, std::vector<int> &cumpulsory_group_vertices, node **host_tree, std::unordered_set<int> &contain_group_vertices)
{
	int bit_num = 1, v;
	for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++, bit_num <<= 1)
	{
		for (size_t to = 0; to < group_graph[*it].size(); to++)
		{
			v = group_graph[*it][to].first;
			host_tree[v][bit_num].cost = 0;
			contain_group_vertices.insert(v);
		}
	}
}
int get_max(int vertex, node **host_tree, int G)
{
	int re = 0;
	for (size_t i = 1; i < G; i <<= 1)
	{
		if (host_tree[vertex][i].cost == 0)
		{
			re += i;
		}
	}

	return re;
}
int graph_v_of_v_idealID_DPBF_vertex_group_set_ID_gpu(int vertex, graph_v_of_v_idealID &group_graph,
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
			ID = ID + pow(2, pow_num);
		}
		pow_num++;
	}

	return ID;
}

__global__ void Relax(queue_element *Queue_dev, int queue_size, queue_element *new_queue_device, int *new_queue_size, int *sets_IDs, int *sets_IDS_pointer, int *edge, int *edge_cost, int *pointer, int width, node *tree, int inf, int *best, int full, int *lb0)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < queue_size)
	{
		queue_element top_node = Queue_dev[idx];
		int v = top_node.v, p = top_node.p, lb = 0;
		int x_slash = full - p, v_line = v * width;
		tree[v_line + p].update = 0;
		if (tree[v_line + x_slash].cost != inf)
		{
			int new_best = tree[v_line + x_slash].cost + tree[v_line + p].cost;
			if (new_best <= (*best))
			{
				atomicMin((int *)best, new_best);
				// atomicMin(&tree[v_line + full].cost, new_best);

				tree[v_line + full].type = 2;
				tree[v_line + full].p1 = p;
				tree[v_line + full].p2 = x_slash;
			}
		}
		else
		{
			int new_best = tree[v_line + p].cost;

			for (size_t i = 1; i <= x_slash; i <<= 1)
			{
				if (i & x_slash)
				{
					new_best += tree[v_line + i].c;
				}
			}
			atomicMin((int *)best, new_best);
		}
		if (tree[v_line + p].cost - 1 > (*best) / 2)
		{
			return;
		}
		for (int i = pointer[v]; i < pointer[v + 1]; i++)
		{
			/*grow*/
			int u = edge[i], cost_euv = edge_cost[i];
			int grow_tree_cost = tree[v_line + p].cost + cost_euv, u_line = u * width;
			int old = atomicMin(&tree[u_line + p].cost, grow_tree_cost);
			if (old > grow_tree_cost)
			{
				tree[u_line + p].type = 1;
				tree[u_line + p].u = v;
				// enqueue operation

				lb = get_lb(tree, u * width, x_slash);
				lb = thrust::max(lb, lb0[u_line + p]);
				if (lb + grow_tree_cost <= (*best))
				{
					int check = atomicCAS(&tree[u_line + p].update, 0, 1);
					if (check == 0)
					{
						int pos = atomicAdd(new_queue_size, 1);
						new_queue_device[pos].v = u;
						new_queue_device[pos].p = p;
					}
				}
			}
		}

		/*merge*/
		for (int it = sets_IDS_pointer[p]; it < sets_IDS_pointer[p + 1]; it++)
		{

			int p2 = sets_IDs[it];
			if (tree[v_line + p2].cost == inf)
			{
				continue;
			}
			int p1_cup_p2 = p + p2;
			int merged_tree_cost = tree[v_line + p].cost + tree[v_line + p2].cost, vpm = v_line + p1_cup_p2;
			int old = atomicMin(&tree[vpm].cost, merged_tree_cost);
			lb = get_lb(tree, v_line, full - p1_cup_p2);
			lb = thrust::max(lb, lb0[v_line + full - p1_cup_p2]);
			if (old > merged_tree_cost)
			{ // O(3^|Gamma||V| comparisons in totel, see the DPBF paper)
				/*update T(v,p1_cup_p2) by merge T(v,p1) with T(v,v2)*/
				tree[vpm].type = 2;
				tree[vpm].p1 = p;
				tree[vpm].p2 = p2;
				if (merged_tree_cost - 1 <= 0.667 * (*best) && lb + merged_tree_cost <= (*best))
				{
					int check = atomicCAS(&tree[vpm].update, 0, 1);
					if (check == 0)
					{
						int pos = atomicAdd(new_queue_size, 1);
						new_queue_device[pos].v = v;
						new_queue_device[pos].p = p1_cup_p2;
					}
				}
			}
		}
	}
}

__global__ void one_label_lb(node *tree, int width, int *lb0, int N, int G, int inf, int *can_find_solution, int *w, int *w1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		int row = idx * width, temp = 0;

		for (size_t i = 1; i < width; i <<= 1)
		{
			if (temp < tree[row + i].c)
			{
				temp = tree[row + i].c;
			}
		}
		if (temp != inf)
		{
			*can_find_solution = 1;
		}
		int vline = idx * width;
		for (int x = 1; x < width; x++)
		{
			lb0[vline + x] = inf;
			for (int i = 0; i < G; i++)
			{
				for (int j = 0; j < G; j++)
				{
					lb0[vline + x] = thrust::min(lb0[vline + x], tree[vline + (1 << i)].c + w[i * width * G + j * width + x] + tree[vline + (1 << j)].c);
				}
			}
			int minj = inf;
			for (size_t i = 0; i < G; i++)
			{
				if ((1 << i) & x)
				{
					minj = thrust::min(minj, tree[idx * width + (1 << i)].c);
				}
			}
			for (size_t i = 0; i < G; i++)
			{
				int vi = 1 << i;
				if (vi & x)
				{
					lb0[vline + x] = thrust::max(lb0[vline + x], tree[vline + (1 << i)].c + w1[i * width + x] + minj);
				}
			}

			lb0[vline + x] /= 2;
		}
	}
}
__global__ void count_set(node *tree,int width,int inf, int N,int *counts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		int vline = idx * width;
		for (int x = 1; x < width; x++)
		{
			if(tree[vline+x].cost!=inf)
			{
			atomicAdd(counts,1);
			}			
		}
	}
}
__global__ void dis0_init_1(queue_element *dis_queue, queue_element *new_dis_queue, node *tree, int *dis_queue_size, int *new_queue_size, int *edge, int *edge_cost, int *pointer, int width)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < *dis_queue_size)
	{
		int u = dis_queue[idx].v, p = dis_queue[idx].p;
		tree[u * width + p].update = 0;

		for (int i = pointer[u]; i < pointer[u + 1]; i++)
		{

			int v = edge[i];
			
			int vp = v * width + p;
			int new_w = tree[u * width + p].c + edge_cost[i];
			int old = atomicMin(&tree[vp].c, new_w);
			if (new_w < old)
			{
				tree[vp].u = u;
				tree[vp].type = 1;
				int check = atomicCAS(&tree[vp].update, 0, 1);
				if (check == 0)
				{
					int pos = atomicAdd(new_queue_size, 1);
					new_dis_queue[pos] = {v, p};
				}
			}
		}
	}
}



void DPBF_GPU(node **host_tree, node *host_tree_one_d, CSR_graph &graph, std::vector<int> &cumpulsory_group_vertices, graph_v_of_v_idealID &group_graph, graph_v_of_v_idealID &input_graph,  int &real_cost, non_overlapped_group_sets s, double &rt,int &RAM,records &ret)
{
	// cudaSetDevice(1);

	auto begin = std::chrono::high_resolution_clock::now();
	auto pbegin = std::chrono::high_resolution_clock::now();
	auto pend = std::chrono::high_resolution_clock::now();
	int width, height, r = 0, process = 0, N = graph.V;
	int *queue_size, *new_queue_size, *lb0, *dis_queue_size, *new_dis_queue_size, *non_overlapped_group_sets_IDs_gpu, *non_overlapped_group_sets_IDs_pointer_device, *can_find;
	int *all_pointer, *all_edge, *edge_cost;
	int *w_d, *w1_d;
	node *tree;
	int *best;
	queue_element *queue_device, *new_queue_device, *new_dis_queue, *dis_queue;
	double time_process = 0;
	int G = cumpulsory_group_vertices.size();
	all_edge = graph.all_edge, all_pointer = graph.all_pointer, edge_cost = graph.all_edge_weight;
	int group_sets_ID_range = pow(2, G) - 1;
	width = group_sets_ID_range + 1, height = N;
	long long unsigned int problem_size = N * pow(2, cumpulsory_group_vertices.size());
	cudaMallocManaged((void **)&can_find, sizeof(int));
	cudaMallocManaged((void **)&dis_queue, N * G * sizeof(queue_element));
	cudaMallocManaged((void **)&new_dis_queue, N * G * sizeof(queue_element));
	cudaMallocManaged((void **)&dis_queue_size, sizeof(int));
	cudaMallocManaged((void **)&new_dis_queue_size, sizeof(int));
	cudaMallocManaged((void **)&queue_size, sizeof(int));
	cudaMallocManaged((void **)&new_queue_size, sizeof(int));
	cudaMallocManaged((void **)&best, sizeof(int));
	cudaMallocManaged((void **)&lb0, problem_size * sizeof(int));
	cudaMallocManaged((void **)&w_d, G * G * width * sizeof(int));
	cudaMallocManaged((void **)&w1_d, G * width * sizeof(int));
	cudaMallocManaged((void **)&queue_device, problem_size * sizeof(queue_element));
	cudaMallocManaged((void **)&new_queue_device, problem_size * sizeof(queue_element));
	cudaMallocManaged((void **)&non_overlapped_group_sets_IDs_gpu, sizeof(int) * s.length);
	cudaMallocManaged((void **)&non_overlapped_group_sets_IDs_pointer_device, sizeof(int) * (group_sets_ID_range + 3));
	cudaMemcpy(non_overlapped_group_sets_IDs_gpu, s.non_overlapped_group_sets_IDs.data(), sizeof(int) * s.length, cudaMemcpyHostToDevice);
	cudaMemcpy(non_overlapped_group_sets_IDs_pointer_device, s.non_overlapped_group_sets_IDs_pointer_host.data(), (group_sets_ID_range + 3) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMallocManaged(&tree, width * height * sizeof(node));
	int vv[G][G] = {inf};
	// host_lb = new int[N];
	int *w = new int[G * G * width], *w1 = new int[G * width];
	int max_queue_size = 0 ;
	// std::cout << "pitch " << pitch_node << " " << " width " << width << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	time_process += runningtime;
	// std ::cout << "main allocate cost time " << runningtime << std ::endl;
	size_t avail(0), total(0);
	cudaMemGetInfo(&avail, &total);
	// std ::cout << "avail " << avail / 1024 / 1024 / 1024 << " total " << total / 1024 / 1024 / 1024 << std ::endl;
	*best = inf;
	*queue_size = 0, *dis_queue_size = 0, *new_queue_size = 0, *new_dis_queue_size = 0;
	begin = std::chrono::high_resolution_clock::now();
	int vnum = input_graph.size() - group_graph.size();
	for(int i=0;i<N;i++)
	{
		for (size_t j = 1; j < width; j++)
		{
			host_tree[i][j].cost = inf;
			host_tree[i][j].c = inf;
		}
		
	}
	std::unordered_set<int> contain_group_vertices;
	set_max_ID(group_graph, cumpulsory_group_vertices, host_tree, contain_group_vertices);
	graph_hash_of_mixed_weighted solution_tree;
	for (auto it = contain_group_vertices.begin(); it != contain_group_vertices.end(); it++)
	{
		int v = *it;
		host_tree[v][0].cost = 0;
		host_tree[v][0].type = 0;
		int group_set_ID_v = get_max(v, host_tree, width); /*time complexity: O(|Gamma|)*/
		for (size_t i = 1; i <= group_set_ID_v; i++)
		{
			if ((i | group_set_ID_v) == group_set_ID_v)
			{
				host_tree[v][i].cost = 0;
				host_tree[v][i].type = 0;

				queue_device[*queue_size].v = v;
				queue_device[*queue_size].p = i;
				*queue_size += 1;
			}
		}
		for (size_t j = 1; j <= group_set_ID_v; j <<= 1)
		{
			if (j & group_set_ID_v)
			{
				host_tree[v][j].c = 0;
				dis_queue[*dis_queue_size].v = v;
				dis_queue[*dis_queue_size].p = j;
				*dis_queue_size += 1;
			}
		}
	}
	cudaMemcpy(tree, host_tree_one_d, width * sizeof(node) * height, cudaMemcpyHostToDevice);
	end = std::chrono::high_resolution_clock::now();
	runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	time_process += runningtime;
	
	begin = std::chrono::high_resolution_clock::now();

	r = 0;
	while (*dis_queue_size != 0)
	{

		// std::cout << "dis round " << r++ << " queue size " << *dis_queue_size << std::endl;
		//  for (size_t i = 0; i < *dis_queue_size; i++)
		//  {
		//  	std::cout << " v " << dis_queue[i].v << " p " << dis_queue[i].p << "; ";
		//  }
		//  cout<<endl;
		dis0_init_1<<<(*dis_queue_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(dis_queue, new_dis_queue, tree, dis_queue_size, new_dis_queue_size, all_edge, edge_cost, all_pointer, width);
		cudaDeviceSynchronize();
		// cudaMemcpy(host_tree_one_d, tree, width * sizeof(node) * height, cudaMemcpyDeviceToHost);
		// for (size_t i = 0; i < height; i++)
		// {
		// 	cout<<i<<" ";
		// 	for (size_t j = 1; j < width; j++)
		// 	{
		// 		cout<<host_tree[i][j].c<<" ";
		// 	}
		// 	cout<<endl;
		// }

		//	std ::cout << "dis cost time " << runningtime << std ::endl;
		*dis_queue_size = *new_dis_queue_size;
		*new_dis_queue_size = 0;
		thrust::swap(dis_queue, new_dis_queue);
	}

	

	cudaMemcpy(host_tree_one_d, tree, width * sizeof(node) * height, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < G; i++)
	{
		for (size_t j = 0; j < G; j++)
		{
			vv[i][j] = inf;
			for (size_t k = 0; k < width; k++)
			{
				w[i * width * G + j * width + k] = inf;
			}
		}
	}

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < G; j++)
		{
			if (host_tree[i][1 << j].c == 0)
			{
				for (size_t k = 0; k < G; k++)
				{
					vv[j][k] = min(vv[j][k], host_tree[i][1 << k].c);
				}
			}
		}
	}
	for (size_t i = 0; i < G; i++)
	{
		//cout << i << " ";
		w[i * width * G + i * width + (1 << i)] = 0;
		for (size_t j = 0; j < G; j++)
		{
		//	cout << vv[i][j] << " ";
			w[i * width * G + j * width] = vv[i][j];
		}
		//cout << endl;
	}
	for (int x = 1; x < width; x++)
	{
		for (int i = 0; i < G; i++)
		{
			for (int j = 0; j < G; j++)
			{
				int xx = x;
				if (xx & (1 << j))
					xx -= (1 << j);
				if (xx & (1 << i))
					xx -= (1 << i);
				if (xx == 0)
				{
					w[i * width * G + j * width + x] = w[i * width * G + j * width];
				}
				for (int p = 0; p < G; p++)
				{
					if ((1 << p) & (xx)) // p作为中间节点的首要条件是p要是xx的一个点
					{
						// printf("x=%d i=%d j=%d mid=%d ,compare %d %d\n", x, i, j, 1 << p, w[i][j][x], w[i][p][xx - (1 << p)] + vv[p][j]);

						w[i * width * G + j * width + x] = min(w[i * width * G + j * width + x], w[i * width * G + p * width + xx - (1 << p)] + vv[p][j]);
					}
				}
				// printf("set  i=%d j=%d, x=%d W=%d\n", i, j, x, w[i][j][x]);
			}
		}
	}
	for (int i = 0; i < G; i++)
	{
		for (size_t x = 1; x < width; x++)
		{
			w1[i * width + x] = inf;
			for (size_t j = 0; j < G; j++)
			{
				w1[i * width + x] = min(w1[i * width + x], w[i * width * G + j * width + x]);
			}
		}
	}

	*can_find = 0;
	cudaMemcpy(w_d, w, G * G * width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(w1_d, w1, G * width * sizeof(int), cudaMemcpyHostToDevice);
	one_label_lb<<<(N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(tree, width, lb0, N, G, inf, can_find, w_d, w1_d);
	cudaDeviceSynchronize();
	// for (size_t i = 0; i < N; i++)
	// {
	// 	cout << i << " lb: " << lb0[i] << " ";
	// }
	// cout<<endl;
	if (*can_find == 0)
	{
		real_cost = 0;
		cout << "can not find a solution" << endl;
		//	return solution_tree;
	}

	//std ::cout << "test cost time " << runningtime << std ::endl;
	// std::cout << "queue size init " << *queue_size << std::endl;
	// std::cout << "queue init " << std::endl;
	/* 	for (size_t i = 0; i < *queue_size; i++)
		{
			std::cout << " v " << queue_device[i].v << " p " << queue_device[i].p << "; ";
		} */

	r = 0;
	pbegin = std::chrono::high_resolution_clock::now();
	while (*queue_size != 0)
	{
		process += *queue_size;
		// std::cout << "round " << r++ << " queue size " << *queue_size << " best " << *best << std::endl;
		// begin = std::chrono::high_resolution_clock::now();
		//  for (size_t i = 0; i < *queue_size; i++)
		//  {
		//  	std::cout << " v " << queue_device[i].v << " p " << queue_device[i].p << "; ";
		//  }
		//  cout<<endl;
		Relax<<<(*queue_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(queue_device, *queue_size, new_queue_device, new_queue_size, non_overlapped_group_sets_IDs_gpu,
																							 non_overlapped_group_sets_IDs_pointer_device, all_edge, edge_cost, all_pointer, width, tree, inf, best, group_sets_ID_range, lb0);
		// cudaMemcpy(host_tree_one_d, tree, width * sizeof(node) * height, cudaMemcpyDeviceToHost);
		// for (size_t i = 0; i < height; i++)
		// {
		// 	cout<<i<<" ";
		// 	for (size_t j = 1; j < width; j++)
		// 	{
		// 		cout<<host_tree[i][j].cost<<" ";
		// 	}
		// 	cout<<endl;
		// }
		cudaDeviceSynchronize();
		max_queue_size = max(max_queue_size,*queue_size);
		// std ::cout << "merge " << *merge_count <<"grow "<<*new_queue_size-*merge_count << std ::endl;
		end = std::chrono::high_resolution_clock::now();
		runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
		// std ::cout << "relax cost time " << runningtime << std ::endl;
		*queue_size = *new_queue_size;
		*new_queue_size = 0;
		// 根据新队列cost和旧队列cost合并新队列和旧队列从运行队列后的元素
		std::swap(queue_device, new_queue_device);
	}
	int *counts;
	cudaMallocManaged((void **)&counts,sizeof(int));
	count_set<<<(N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(tree,width,inf,N,counts);
	cudaDeviceSynchronize();
	RAM = (*counts+max_queue_size+N*width);
	//cout<<"total process node "<<process<<endl;
	ret.counts = *counts;
	ret.process_queue_num = process;
	pend = std::chrono::high_resolution_clock::now();
	runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(pend - pbegin).count() / 1e9; // s
	rt = runningtime;
	
	begin = std::chrono::high_resolution_clock::now();
	cudaMemcpy(host_tree_one_d, tree, width * sizeof(node) * height, cudaMemcpyDeviceToHost);
	int min_cost = inf, min_node = -1;
	for (int i = 0; i < N; i++)
	{
		// cout << host_tree[i][group_sets_ID_range].cost << " ";
		if (host_tree[i][group_sets_ID_range].cost < min_cost)
		{
			min_cost = host_tree[i][group_sets_ID_range].cost;
			min_node = i;
		}
	}
	real_cost = min_cost;
	//std::cout << "gpu cost " << min_cost << " root at " << min_node << std::endl;
	if (min_node == -1)
	{real_cost = 0;
		cout<<"can not find sulition"<<endl;
		// return solution_tree;
	}

	// std::queue<std::pair<int, int>> waited_to_processed_trees; // <v, p>
	// int root_v = min_node, root_p = group_sets_ID_range;
	// waited_to_processed_trees.push({root_v, root_p});
	// r = 0;
	// while (waited_to_processed_trees.size() > 0&&++r<200)
	// {

	// 	int v = waited_to_processed_trees.front().first, p = waited_to_processed_trees.front().second;
	// 	waited_to_processed_trees.pop();
	// 	graph_hash_of_mixed_weighted_add_vertex(solution_tree, v, 0);
	// 	auto pointer_trees_v_p = host_tree[v][p];
	// 	int form_type = pointer_trees_v_p.type;
	// 	if (form_type == 0)
	// 	{ // T(v,p) is a single vertex
	// 	}
	// 	else if (form_type == 1)
	// 	{ // T(v,p) is formed by grow
	// 		int u = host_tree[v][p].u;
	// 		waited_to_processed_trees.push({u, p});
	// 		/*insert (u,v); no need to insert weight of u here, which will be inserted later for T(u,p)*/
	// 		int c_uv = graph_v_of_v_idealID_edge_weight(input_graph, u, v);
	// 		graph_hash_of_mixed_weighted_add_edge(solution_tree, u, v, c_uv);
	// 	}
	// 	else
	// 	{ // T(v,p) is formed by merge
	// 		int p1 = host_tree[v][p].p1, p2 = host_tree[v][p].p2;
	// 		waited_to_processed_trees.push({v, p1});
	// 		waited_to_processed_trees.push({v, p2});
	// 	}
	// }

	cudaFree(tree);
	cudaFree(lb0);
	cudaFree(queue_device);
	cudaFree(new_queue_device);
	cudaFree(dis_queue);
	cudaFree(new_dis_queue);
	cudaFree(w1_d);
	cudaFree(w_d);
	cudaFree(non_overlapped_group_sets_IDs_gpu);
	cudaFree(non_overlapped_group_sets_IDs_pointer_device);
	cudaFree(best);
	end = std::chrono::high_resolution_clock::now();
	runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

	// std ::cout << "form tree cost time " << runningtime << std ::endl;
	//	return solution_tree;
}
