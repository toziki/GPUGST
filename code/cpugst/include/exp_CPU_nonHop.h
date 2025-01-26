#pragma once

#include <chrono>
#include <queue>
#include <omp.h>
#include <boost/heap/fibonacci_heap.hpp>
#include <graph_hash_of_mixed_weighted/graph_hash_of_mixed_weighted.h>
#include <graph_hash_of_mixed_weighted/common_algorithms/graph_hash_of_mixed_weighted_connected_components.h>

#include <graph_hash_of_mixed_weighted/two_graphs_operations/graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID.h>

#include <graph_hash_of_mixed_weighted_read_for_GSTP.h>
#include "graph_hash_of_mixed_weighted_sum_of_nw_ec.h"
#include "CPUNONHOP.h"
#include <future>
using namespace std;
bool this_is_a_feasible_solution_gpu(graph_hash_of_mixed_weighted &solu, graph_v_of_v_idealID &group_graph,
									 std::vector<int> &group_vertices)
{

	/*time complexity O(|V_solu|+|E_solu|)*/
	if (graph_hash_of_mixed_weighted_connected_components(solu).size() != 1)
	{ // it's not connected
		std::cout << "this_is_a_feasible_solution: solu is disconnected!" << endl;
		return false;
	}

	for (auto it = group_vertices.begin(); it != group_vertices.end(); it++)
	{
		int g = *it;
		bool covered = false;
		for (auto it2 = solu.hash_of_vectors.begin(); it2 != solu.hash_of_vectors.end(); it2++)
		{
			int v = it2->first;
			if (graph_v_of_v_idealID_contain_edge(group_graph, g, v))
			{
				covered = true;
				break;
			}
		}
		if (covered == false)
		{
			std::cout << "this_is_a_feasible_solution: a group is not covered!" << endl;
			return false;
		}
	}

	return true;
}
void exp_CPU_nonHop(string path, string data_name, int T, int task_start_num, int task_end_num)
{
	
	std::cout << "start.. " << std::endl;

	std::vector<int> generated_group_vertices;
	std::unordered_set<int> generated_group_vertices_hash;
	std::vector<std::vector<int>> inquire;
	graph_v_of_v_idealID v_generated_group_graph, v_instance_graph;

	int ov = read_input_graph(path + data_name + ".in", v_instance_graph);

	int V = v_instance_graph.size();
	std::cout << "read input complete" << endl;
	read_Group(path + data_name + ".g", v_instance_graph, v_generated_group_graph);

	std::cout << "read group complete " << v_generated_group_graph.size() << endl;
	std::cout << "enter " << endl;

	read_inquire(path + data_name  + to_string(T) + ".csv", inquire);
	int iteration_times = inquire.size();
	std::cout << "inquires size " << inquire.size() << " G = " << inquire[0].size() << endl;
	int group_sets_ID_range = pow(2, T) - 1;
	/*iteration*/
	std::cout << "------------------------------------------------------------" << endl;
	int rounds = 0, cpu_cost, gpu_cost;

	/*output*/
	ofstream outputFile;
	
	outputFile.precision(8);
	outputFile.setf(ios::fixed);
	outputFile.setf(ios::showpoint);
	outputFile.open(path+"result/exp_CPU_nonHop_" + data_name + "_T" + to_string(T) + "_" + to_string(task_start_num) + "-" + to_string(task_end_num) + ".csv");
	outputFile << "task_ID,task,CPU_nonHop_time,CPU_nonHop_cost,CPU_nonHop_memory,counts,process_num" << endl;
	std::cout << task_start_num<<" "<<task_end_num << endl;
	for (int i = task_start_num; i <= task_end_num; i++)
	{
		rounds++;

		std::cout << data_name << " iteration " << i << endl;

		string task = "";
		
		generated_group_vertices.clear();
		generated_group_vertices_hash.clear();
		for (size_t j = 0; j < inquire[i].size(); j++)
		{
			generated_group_vertices.push_back(inquire[i][j]);
			generated_group_vertices_hash.insert(inquire[i][j]);
	
		
			task += to_string(inquire[i][j]) + " ";
		}
	
		if (1)
		{

			
				int RAM = 0;
				auto begin = std::chrono::high_resolution_clock::now();
				double time_record = 0;
				records ret;
				graph_hash_of_mixed_weighted solu = graph_v_of_v_idealID_PrunedDPPlusPlus(v_instance_graph, v_generated_group_graph, generated_group_vertices_hash, 1, RAM, time_record,ret);
				auto end = std::chrono::high_resolution_clock::now();
				double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
				
				cpu_cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);
				printf("cost=%d\n",cpu_cost);
				//graph_hash_of_mixed_weighted_print_size(solu);
				if (!this_is_a_feasible_solution_gpu(solu, v_generated_group_graph, generated_group_vertices))
				{
					std::cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
					graph_hash_of_mixed_weighted_print(solu);
					
					cpu_cost = 100000;//can not find solution
				}
				std::cout << "------------------------------------------------------------" << endl;
				
				outputFile << i << "," << task << "," << time_record << "," << cpu_cost << "," << RAM<<","<<ret.counts<<"," <<ret.process_queue_num<< endl;

			

		}
	}

	outputFile << endl;
}
