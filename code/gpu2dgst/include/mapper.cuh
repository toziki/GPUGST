#ifndef __MAPPER__
#define __MAPPER__
#include "header.h"
#include "util.h"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include <thrust/detail/minmax.h>
#include <assert.h>
typedef feature_t (*cb_mapper)(vertex_t active_edge_src,
							   vertex_t active_edge_end,
							   feature_t level,
							   index_t *beg_pos,
							   weight_t weight_list,
							   feature_t *vert_status,
							   feature_t *vert_status_prev);

// Bitmap could also be helpful
/* mapper kernel function */
class mapper
{
public:
	// variable
	vertex_t *adj_list;
	weight_t *weight_list;
	index_t *beg_pos;
	index_t vert_count;
	feature_t *vert_status;
	feature_t *vert_status_prev;
	feature_t *one_label_lower_bound;
	feature_t *lb_record;
	feature_t *merge_or_grow;
	int width, full, *lb0, diameter;
	int val1, val2;
	volatile int *best;
	int *merge_pointer;
	int *merge_groups;
	// index_t *cat_thd_count_sml;
	cb_mapper edge_compute;
	cb_mapper edge_compute_push;
	cb_mapper edge_compute_pull;

public:
	// constructor
	mapper(gpu_graph ggraph, meta_data mdata,
		   cb_mapper user_mapper_push,
		   cb_mapper user_mapper_pull,
		   int wid)
	{
		adj_list = ggraph.adj_list;
		weight_list = ggraph.weight_list;
		beg_pos = ggraph.beg_pos;
		width = wid;
		best = mdata.best;
		full = width - 1;
		diameter = mdata.diameter;
		val1 = width * (diameter + 1), val2 = (diameter + 1);
		merge_groups = ggraph.merge_groups_d;
		merge_pointer = ggraph.merge_pointer_d;
		// 		for (size_t i = 0; i < width; i++)
		// {
		// 	cout<<i<<"  "<<"begin "<< merge_pointer[i] <<" end "<< merge_pointer[i+1]<<" ";
		// 	for (size_t j = merge_pointer[i]; j <merge_pointer[i+1]; j++)
		// 	{
		// 		cout<<	merge_groups[j]<<" ";
		// 	}
		// 	cout<<endl;
		// }
		cout << "end " << merge_pointer[width];

		vert_count = ggraph.vert_count;
		vert_status = mdata.vert_status;
		vert_status_prev = mdata.vert_status_prev;
		// cat_thd_count_sml = mdata.cat_thd_count_sml;

		edge_compute_push = user_mapper_push;
		edge_compute_pull = user_mapper_pull;
	}

	~mapper() {}

public:
	// function
	// Could represent thread, warp, cta, grid grained thread scheduling

	__forceinline__ __device__ void
	mapper_push(		 // 这个是大中小都使用的push函数 这里会进行更新新的距离
		vertex_t wqueue, // 这个是本轮的用值
		vertex_t *worklist,
		index_t *cat_thd_count, // 这个是本轮返回值
		const index_t GRP_ID,
		const index_t GRP_SZ,
		const index_t GRP_COUNT,
		const index_t THD_OFF,
		feature_t level)
	{
		index_t appr_work = 0;
		weight_t weight;
		const vertex_t WSZ = wqueue;
		int mxid = val1 * (vert_count - 1) + val2 * (width - 1) + diameter;
		for (index_t i = GRP_ID; i < WSZ; i += GRP_COUNT)
		{
			vertex_t frontier = worklist[i];
			int v = frontier / val1, d = frontier % val2;

			int p = (frontier - v * val1 - d) / val2;
			index_t beg = beg_pos[v], end = beg_pos[v + 1];
			if (d < diameter)
			{
				for (index_t j = beg + THD_OFF; j < end; j += GRP_SZ)
				{
					// printf("");
					vertex_t vert_end = adj_list[j];
					weight = weight_list[j];
					vertex_t update_dest = vert_end * val1 + p * val2  +d+ 1;
					feature_t dist = (*edge_compute_push)(frontier, update_dest,
														  level, beg_pos, weight, vert_status, vert_status_prev); // 获取值 根据任务选择操作
					if (vert_status[update_dest] > dist)
					{
						
						atomicMin(vert_status + update_dest, dist);
					}
				}
			}

			// merge
			beg = merge_pointer[p], end = merge_pointer[p + 1];
			int dist;
			for (index_t j = beg + THD_OFF; j < end; j += GRP_SZ)
			{	
				int p_comp = merge_groups[j];
				for (int dia = 0; dia <= diameter - d; dia++)
				{
					// printf("m");
					// if (v * val1 + p_comp * val2 + dia > mxid || v * val1 + p_comp * val2 + dia < 0) {
					// 	printf("1222");
					// }
					weight_t weight = vert_status[v * val1 + p_comp * val2 + dia];
					int new_d = dia>d?dia:d;
					
					vertex_t update_dest = v * val1 + (p_comp | p) * val2 + new_d;
					// if (update_dest > mxid || update_dest < 0) {
					// 	printf("1234");
					// }
					if (frontier < 0 || frontier > mxid) {
						printf("123");
					}
					dist = vert_status[frontier] + weight;
						// dist = (*edge_compute_push)(frontier, update_dest,//极其奇怪的bug 这个函数不能替换成上面语句？
						// 							level, beg_pos, weight, vert_status, vert_status_prev); // 获取值 根据任务选择操作
					if (vert_status[update_dest] > dist)
					{
						
						atomicMin(vert_status + update_dest, dist);
					}
				}
			}
		}

		// note, we use cat_thd_count to store the future amount of workload
		// and such data is important for switching between push - pull models.
		cat_thd_count[threadIdx.x + blockIdx.x * blockDim.x] = appr_work;
	}

	

	


};

#endif
