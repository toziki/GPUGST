#ifndef _ENACTOR_H_
#define _ENACTOR_H_
#include "gpu_graph.cuh"
#include "meta_data.cuh"

#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include <limits.h>
#include "barrier.cuh"


__global__ void
balanced_push_kernel(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{

	// 从CPU函数传递来的第一层核函数
	// 核函数 这里的上一层已经调用起全部线程了
	//__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride

	feature_t level_thd = level[0]; // 可能就是只用了0？？ while循环的次数
	

	if (!TID)
	{level[6]=0;
		mdata.worklist_sz_mid[0] = 0; // 如果TID是0 把全局的中等大小列表置0 那这里最初就必须全部放在mid了
		
	}

	global_barrier.sync_grid_opt();
	// 下面这个函数扫描了线程对应的点 加入了工作队列
	//  所有线程共同做上面这一项任务 取出来的任务数量也是一样的
	int r=0;
	while (true&&++r<20) // 这里还是一个循环任务 想要一次做完 仅当工作量到阈值时切换出去
	{
		// 平衡
		mdata.future_work[0] = 0;
		global_barrier.sync_grid_opt();

		if (!TID)
		{	
			 int tot = mdata.worklist_sz_mid[0]+mdata.worklist_sz_sml[0]+mdata.worklist_sz_lrg[0];
			level[6]+=tot;
			mdata.worklist_sz_mid[0] = 0; // 下面应该是在扫描执行了 先把新的队列长度置0
			mdata.worklist_sz_sml[0] = 0; // indicate whether bin overflow
			mdata.worklist_sz_lrg[0] = 0;
			if(tot>level[5])
			level[5] = tot;

		}
		// worklist_gather._push_coalesced_scan_random_list(TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd+1);
		worklist_gather._push_coalesced_scan_random_list(TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd + 1);
		// compute on the graph  在图上计算并且立刻形成工作队列
		// and generate frontiers immediately
		global_barrier.sync_grid_opt();
		if (mdata.worklist_sz_sml[0] +
				mdata.worklist_sz_mid[0] +
				mdata.worklist_sz_lrg[0] ==
			0)
			break;
		// global_barrier.sync_grid_opt();

		// Three push mappers.

		compute_mapper.mapper_push(
			mdata.worklist_sz_lrg[0],
			mdata.worklist_lrg,
			mdata.cat_thd_count_lrg,
			blockIdx.x,	 /*group id*/
			blockDim.x,	 /*group size*/
			gridDim.x,	 /*group count*/
			threadIdx.x, /*thread off intra group*/
			level_thd);

		compute_mapper.mapper_push(
			mdata.worklist_sz_mid[0],
			mdata.worklist_mid,
			mdata.cat_thd_count_mid,

			wid_in_grd, /*group id*/
			32,			/*group size*/
			WGRNTY,		/*group count*/
			tid_in_wrp, /*thread off intra group*/
			level_thd);

		compute_mapper.mapper_push(
			mdata.worklist_sz_sml[0],
			mdata.worklist_sml,
			mdata.cat_thd_count_sml,
			TID,   /*group id*/
			1,	   /*group size*/
			GRNTY, /*group count*/
			0,	   /*thread off intra group*/
			level_thd);

		//      global_barrier.sync_grid_opt();

		_grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] +
										 mdata.cat_thd_count_mid[TID] +
										 mdata.cat_thd_count_lrg[TID],
									 mdata.future_work);

		global_barrier.sync_grid_opt();
		if (mdata.future_work[0] > ggraph.edge_count * SWITCH_TO && 0)
		{ // pull和剪枝技术的兼容应该是问题 暂时不考虑

			break;
		}
#ifdef ENABLE_MONITORING
		if (!TID)
			printf("level-%d-futurework: %d\n", (int)level, mdata.future_work[0]);
#endif
		// if(level == 1) break;
	}

	if (!TID)
		level[0] = level_thd;
}




int balanced_push(
	int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier)
{
	// 分成三种情况大中小 现在要改的函数！！！
	int blk_size = 0;
	int grd_size = 0;
	// cudaFuncGetAttributes
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size,
									   balanced_push_kernel, 0, 0);

	grd_size = (blk_size * grd_size) / cfg_blk_size;
	blk_size = cfg_blk_size;
	// grd_size = (blk_size * grd_size)/ 128;
	// blk_size = 128;

	//printf("balanced push-- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size * grd_size <= BLKS_NUM * THDS_NUM);

	// push_pull_opt_kernel
	double time = wtime();
	balanced_push_kernel<<<grd_size, blk_size>>>(level,
												 ggraph,
												 mdata,
												 compute_mapper,
												 worklist_gather,
												 global_barrier);

	H_ERR(cudaDeviceSynchronize());
	
	
	// cudaMemcpy(mdata.sml_count_chk, mdata.cat_thd_count_sml, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(mdata.mid_count_chk, mdata.cat_thd_count_mid, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(mdata.lrg_count_chk, mdata.cat_thd_count_lrg, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);

	// index_t total_count = 0;
	// for(int i = 0; i < blk_size * grd_size; i++)
	//     total_count+=mdata.sml_count_chk[i] + mdata.mid_count_chk[i] + mdata.lrg_count_chk[i];

	// printf("---debug total count: %ld\n", total_count);
	return 0;
}






#endif
