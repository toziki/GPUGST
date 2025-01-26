#ifndef _ENACTOR_H_
#define _ENACTOR_H_
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include "reducer_enactor.cuh"
#include <thrust/swap.h>
#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include <limits.h>
#include "barrier.cuh"


// Push model: one kernel for multiple iterations
__global__ void
hybrid_bin_scan_push_kernel(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,	 // 使用工作队列更新
	reducer worklist_gather, // 收集工作队列
	Barrier global_barrier)
{ // 从CPU函数传递来的第一层核函数
	// 核函数 这里的上一层已经调用起全部线程了
	__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride
	const index_t BIN_OFF = TID * BIN_SZ;		   // 规定了一个线程的binsize是32

	feature_t level_thd = level[0];
	vertex_t output_off;

	// vertex_t mdata.mdata.worklist_bin_reg[16];
	// Not a big difference comparing to
	// directly store frontiers in global mem
	if (!TID)
		mdata.worklist_sz_mid[0] = 0; // 如果TID是0 把全局的中等大小列表置0 那这里最初就必须全部放在mid了

	global_barrier.sync_grid_opt();																								 // 下面这个函数扫描了线程对应的点 加入了工作队列
	worklist_gather._push_coalesced_scan_single_random_list(smem, TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd); // 这里生成工作队列
	// 所有线程共同做上面这一项任务 取出来的任务数量也是一样的

	vertex_t mid_queue = mdata.worklist_sz_mid[0]; // 好像就是取了指针* 这里就是size
	level[1] = mid_queue;
	// printf("at level %d qsize %d\n",level_thd,mid_queue);
#ifdef ENABLE_MONITORING
	if (!TID)
		printf("Entering-hybrid-bin-front-count: %d\n", mid_queue);
#endif

	while (true) // 这里还是一个循环任务 想要一次做完 仅当工作量到阈值时切换出去
	{
		/* CANNOT immediately change worklist_sz_sml[0] to 0
		 * global_barrier could sync threads, not memory updates
		 * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
		 * should be far away after
		 * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
		 */
		mdata.future_work[0] = 0;
		global_barrier.sync_grid_opt();

		if (!TID)
		{
			mdata.worklist_sz_mid[0] = 0; // 下面应该是在扫描执行了 先把新的队列长度置0
			mdata.worklist_sz_sml[0] = 0; // indicate whether bin overflow
		}
		vertex_t my_front_count = 0;
		vertex_t bests = *(mdata.best);
		// compute on the graph  在图上计算并且立刻形成工作队列
		// and generate frontiers immediately
		global_barrier.sync_grid_opt();

#ifdef ENABLE_MONITORING
		if (!TID)
			printf("level-%d-frontier-count: %d\n", (int)level_thd, mid_queue);
#endif
		index_t appr_work = 0;

		// Online filter is included.
		//-Comment out recoder to disable online filter.
		// 在线过滤
		compute_mapper.mapper_bin_push(
			appr_work, // 点上更新后产生的新任务数量
			mdata.worklist_sz_sml,
			my_front_count,
			mdata.worklist_bin,
			mid_queue,
			mdata.worklist_mid,
			wid_in_grd, /*group id*/
			32,			/*group size*/
			WGRNTY,		/*group count*/
			tid_in_wrp, /*thread off intra group*/
			level_thd,
			BIN_OFF, mdata.best, mdata.record, mdata.lb_record);

		// global_barrier.sync_grid_opt();

		_grid_sum<vertex_t, index_t>(appr_work, mdata.future_work); // future是一个计数值 appwork是工作队列点上的工作量和
		global_barrier.sync_grid_opt();
		if (mdata.future_work[0] > ggraph.edge_count * SWITCH_TO && 0)
		{
#ifdef ENABLE_MONITORING
			if (!TID)
				printf("----->>>Switch to pull model<<<-----------\n");
#endif
			level[2] = mdata.future_work[0];
			level[3] = 555;
			break; // 这里怎么对SSSP这种任务也是如此操作？ 当下阶段工作数量大于某个阈值 就切换了工作模式
		}
		// worklist_gather._push_coalesced_scan_single_random_list
		//     (smem,TID, wid_in_blk, tid_in_wrp,wcount_in_blk,GRNTY,level_thd+1);

		if (mdata.worklist_sz_sml[0] == -1) // means overflow 溢出才重新更新一次
		// if(true)// - Intentionally always overflow, for the purpose of test online filter overhead.
		{ // 如果有某个节点发生溢出 那就需要重新组织全局工作队列
#ifdef ENABLE_MONITORING
			if (!TID)
				printf("------->>>Switch to Ballot filtering<<<-------\n");
#endif

			worklist_gather._push_coalesced_scan_single_random_list(smem, TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd + 1);
		}
		else
		{ // 如果没有节点溢出 那么可以直接从bin中的任务通过前缀和计算得到下一轮工作队列
#ifdef ENABLE_MONITORING
			if (!TID)
				printf("--->>>>>>Online-Filter<<<<<<-----------\n");
#endif
			// Attention, its likely frontier list size goes beyond vert_count
			_grid_scan<vertex_t, vertex_t>(tid_in_wrp,
										   wid_in_blk,
										   wcount_in_blk,
										   my_front_count,
										   output_off,
										   smem,
										   mdata.worklist_sz_mid);

			// compact all thread bins in frontier queue
			worklist_gather._thread_stride_gather(mdata.worklist_mid,
												  mdata.worklist_bin,
												  my_front_count,
												  output_off,
												  BIN_OFF);
		}

		global_barrier.sync_grid_opt();
		if ((mid_queue = mdata.worklist_sz_mid[0]) == 0)
			break; //||退出条件 当中等队列为空
#ifndef __VOTE__
		for (index_t i = TID; i < ggraph.vert_count; i += GRNTY)
			mdata.vert_status_prev[i] = mdata.vert_status[i];
#endif
		level_thd++;

		// if (level_thd == 20) break;//调试用
	}
	if (!TID)
		level[0] = level_thd;
}

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
	{
		mdata.best[0] = inf;
		level[5] = 0;
		level[6] = 0;
		mdata.worklist_sz_mid[0] = 0; // 如果TID是0 把全局的中等大小列表置0 那这里最初就必须全部放在mid了
	}

	global_barrier.sync_grid_opt();
	// 下面这个函数扫描了线程对应的点 加入了工作队列
	//  所有线程共同做上面这一项任务 取出来的任务数量也是一样的

	while (true) // 这里还是一个循环任务 想要一次做完 仅当工作量到阈值时切换出去
	{
		/* CANNOT immediately change worklist_sz_sml[0] to 0
		 * global_barrier could sync threads, not memory updates
		 * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
		 * should be far away after
		 * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
		 */
		// 平衡
		if (!TID)
		{   
			mdata.worklist_sz_mid[0] = 0; // 下面应该是在扫描执行了 先把新的队列长度置0
			mdata.worklist_sz_sml[0] = 0; // indicate whether bin overflow
			mdata.worklist_sz_lrg[0] = 0;

		}
		// worklist_gather._push_coalesced_scan_random_list(TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd+1);
		worklist_gather._push_coalesced_scan_random_list_best(TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd + 1, mdata.record);
		// compute on the graph  在图上计算并且立刻形成工作队列
		// and generate frontiers immediately
		global_barrier.sync_grid_opt();

		if (mdata.worklist_sz_sml[0] +
				mdata.worklist_sz_mid[0] +
				mdata.worklist_sz_lrg[0] ==
			0)
			break;
		if (!TID)
		{   int tot = mdata.worklist_sz_sml[0] +
				mdata.worklist_sz_mid[0] +
				mdata.worklist_sz_lrg[0];
				level[6] += tot;
			if(tot>level[5])
			level[5] = tot;
		
		}
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
			level_thd, mdata.best, mdata.lb_record);

		compute_mapper.mapper_push(
			mdata.worklist_sz_mid[0],
			mdata.worklist_mid,
			mdata.cat_thd_count_mid,

			wid_in_grd, /*group id*/
			32,			/*group size*/
			WGRNTY,		/*group count*/
			tid_in_wrp, /*thread off intra group*/
			level_thd, mdata.best, mdata.lb_record);

		compute_mapper.mapper_push(
			mdata.worklist_sz_sml[0],
			mdata.worklist_sml,
			mdata.cat_thd_count_sml,
			TID,   /*group id*/
			1,	   /*group size*/
			GRNTY, /*group count*/
			0,	   /*thread off intra group*/
			level_thd, mdata.best, mdata.lb_record);

		//      global_barrier.sync_grid_opt();

		_grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] +
										 mdata.cat_thd_count_mid[TID] +
										 mdata.cat_thd_count_lrg[TID],
									 mdata.future_work);

		global_barrier.sync_grid_opt();
	}
}

__global__ void
hybrid_bin_scan_push_kernel_only(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,	 // 使用工作队列更新
	reducer worklist_gather, // 收集工作队列
	Barrier global_barrier)
{
	// 核函数 这里的上一层已经调用起全部线程了
	__shared__ vertex_t smem[32]; // 32位共享内存
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;   // 线程在warp的位置
	const index_t wid_in_blk = threadIdx.x >> 5;   // warp在block的位置
	const index_t wid_in_grd = TID >> 5;		   // warp在grid的位置
	const index_t wcount_in_blk = blockDim.x >> 5; // 一个block里面有多少warp
	const index_t WGRNTY = GRNTY >> 5;			   // warp stride
	const index_t BIN_OFF = TID * BIN_SZ;		   // 规定了一个线程的binsize是32
	global_barrier.sync_grid_opt();
	feature_t level_thd = level[0]; // 可能就是只用了0？？ while循环的次数
	vertex_t output_off;

	// vertex_t mdata.mdata.worklist_bin_reg[16];
	// Not a big difference comparing to
	// directly store frontiers in global mem
	if (!TID)
		mdata.worklist_sz_mid[0] = 0; // 如果TID是0 把全局的中等大小列表置0 那这里最初就必须全部放在mid了

	global_barrier.sync_grid_opt();																								 // 下面这个函数扫描了线程对应的点 加入了工作队列
	worklist_gather._push_coalesced_scan_single_random_list(smem, TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd); // 这里生成工作队列
	// 所有线程共同做上面这一项任务 取出来的任务数量也是一样的
	vertex_t mid_queue = mdata.worklist_sz_mid[0]; // 好像就是取了指针* 这里就是size
	level[1] = mid_queue;
	// printf("at level %d qsize %d\n",level_thd,mid_queue);
#ifdef ENABLE_MONITORING
	if (!TID)
		printf("Entering-hybrid-bin-front-count: %d\n", mid_queue);
#endif

	while (true) // 这里还是一个循环任务 想要一次做完 仅当工作量到阈值时切换出去
	{
		/* CANNOT immediately change worklist_sz_sml[0] to 0
		 * global_barrier could sync threads, not memory updates
		 * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
		 * should be far away after
		 * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
		 */
		mdata.future_work[0] = 0;
		global_barrier.sync_grid_opt();

		if (!TID)
		{
			mdata.worklist_sz_mid[0] = 0; // 下面应该是在扫描执行了 先把新的队列长度置0
			mdata.worklist_sz_sml[0] = 0; // indicate whether bin overflow
		}
		vertex_t my_front_count = 0;

		// compute on the graph  在图上计算并且立刻形成工作队列
		// and generate frontiers immediately
		global_barrier.sync_grid_opt();
#ifdef ENABLE_MONITORING
		if (!TID)
			printf("level-%d-frontier-count: %d\n", (int)level_thd, mid_queue);
#endif
		index_t appr_work = 0;

		// Online filter is included.
		//-Comment out recoder to disable online filter.
		// 在线过滤
		compute_mapper.mapper_bin_push_only(
			appr_work, // 点上更新后产生的新任务数量
			mdata.worklist_sz_sml,
			my_front_count,
			mdata.worklist_bin,
			mid_queue,
			mdata.worklist_mid,
			wid_in_grd, /*group id*/
			32,			/*group size*/
			WGRNTY,		/*group count*/
			tid_in_wrp, /*thread off intra group*/
			level_thd,
			BIN_OFF);

		// global_barrier.sync_grid_opt();

		_grid_sum<vertex_t, index_t>(appr_work, mdata.future_work); // future是一个计数值 appwork是工作队列点上的工作量和
		global_barrier.sync_grid_opt();
		if (mdata.future_work[0] > ggraph.edge_count * SWITCH_TO && 0)
		{
#ifdef ENABLE_MONITORING
			if (!TID)
				printf("----->>>Switch to pull model<<<-----------\n");
#endif
			level[2] = mdata.future_work[0];
			level[3] = 555;
			break; // 这里怎么对SSSP这种任务也是如此操作？ 当下阶段工作数量大于某个阈值 就切换了工作模式
		}
		// worklist_gather._push_coalesced_scan_single_random_list
		//     (smem,TID, wid_in_blk, tid_in_wrp,wcount_in_blk,GRNTY,level_thd+1);

		if (mdata.worklist_sz_sml[0] == -1) // means overflow 溢出才重新更新一次
		// if(true)// - Intentionally always overflow, for the purpose of test online filter overhead.
		{ // 溢出的时候使用ballot 准确算出队列
#ifdef ENABLE_MONITORING
			if (!TID)
				printf("------->>>Switch to Ballot filtering<<<-------\n");
#endif

			worklist_gather._push_coalesced_scan_single_random_list(smem, TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd + 1);
		}
		else
		{ // 没有溢出的时候使用就地利用 也就是online
#ifdef ENABLE_MONITORING
			if (!TID)
				printf("--->>>>>>Online-Filter<<<<<<-----------\n");
#endif
			// Attention, its likely frontier list size goes beyond vert_count
			_grid_scan<vertex_t, vertex_t>(tid_in_wrp,
										   wid_in_blk,
										   wcount_in_blk,
										   my_front_count,
										   output_off,
										   smem,
										   mdata.worklist_sz_mid);

			// compact all thread bins in frontier queue
			worklist_gather._thread_stride_gather(mdata.worklist_mid,
												  mdata.worklist_bin,
												  my_front_count,
												  output_off,
												  BIN_OFF);
		}

		global_barrier.sync_grid_opt();
		if ((mid_queue = mdata.worklist_sz_mid[0]) == 0)
			break; //||退出条件 当中等队列为空
#ifndef __VOTE__
		for (index_t i = TID; i < ggraph.vert_count; i += GRNTY)
			mdata.vert_status_prev[i] = mdata.vert_status[i];
#endif
		level_thd++;
	}
	if (!TID)
		level[0] = level_thd;
}




__global__ void
mapper_bin_push_kernel(
	feature_t level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper)
{

	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	// const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wid_in_grd = TID >> 5;
	// const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	const index_t BIN_OFF = TID * BIN_SZ;
	if (TID == 0)
		mdata.worklist_sz_sml[0] = 0;
	vertex_t bests = *(mdata.best);
	vertex_t my_bin_sz = 0;
	index_t appr_work = 0;
	compute_mapper.mapper_bin_push(
		appr_work,
		mdata.worklist_sz_sml,
		my_bin_sz,
		mdata.worklist_bin,
		mdata.worklist_sz_mid[0],
		mdata.worklist_mid,
		wid_in_grd,
		32,
		WGRNTY,
		tid_in_wrp,
		level,
		BIN_OFF, mdata.best, mdata.record, mdata.lb_record);

	assert(mdata.worklist_sz_sml[0] != -1);

	mdata.cat_thd_count_mid[TID] = my_bin_sz;
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


int mapper_hybrid_push_merge(
	int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier, int init_flag)
{
	// CPU函数
	int blk_size = 0;
	int grd_size = 0;
	int best_h = inf;
	cudaMemcpy((int *)mdata.best, &best_h, sizeof(int), cudaMemcpyHostToDevice);
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size,
									   hybrid_bin_scan_push_kernel, 0, 0); // 库函数 获取的是指定值

	grd_size = (blk_size * grd_size) / cfg_blk_size;
	blk_size = cfg_blk_size;

	// printf("merge -- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size * grd_size <= BLKS_NUM * THDS_NUM);
	if (init_flag == 1)
	{
		hybrid_bin_scan_push_kernel_only<<<grd_size, blk_size>>>(level,
																 ggraph,
																 mdata,
																 compute_mapper,
																 worklist_gather,
																 global_barrier);
	}
	else
	{
		hybrid_bin_scan_push_kernel<<<grd_size, blk_size>>>(level,
															ggraph,
															mdata,
															compute_mapper,
															worklist_gather,
															global_barrier);
	}

	cudaMemcpy(&best_h, (int *)mdata.best, sizeof(int), cudaMemcpyDeviceToHost);
	H_ERR(cudaDeviceSynchronize());
	return 0;
}



#endif
