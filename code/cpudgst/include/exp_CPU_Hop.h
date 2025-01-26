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
#include "CPUHOP.h"
#include <future>
typedef struct non_overlapped_group_sets{
	std::vector<int> non_overlapped_group_sets_IDs_pointer_host, non_overlapped_group_sets_IDs;
	int length;
}non_overlapped_group_sets;
non_overlapped_group_sets graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(int group_sets_ID_range)
{
	non_overlapped_group_sets s;
	s.length = 0;
	s.non_overlapped_group_sets_IDs_pointer_host.resize(group_sets_ID_range + 3);
	/*this function calculate the non-empty and non_overlapped_group_sets_IDs of each non-empty group_set ID;
	time complexity: O(4^|Gamma|), since group_sets_ID_range=2^|Gamma|;
	the original DPBF code use the same method in this function, and thus has the same O(4^|Gamma|) complexity;*/
	// <set_ID, non_overlapped_group_sets_IDs>
	for (int i = 1; i <= group_sets_ID_range; i++)
	{ // i is a nonempty group_set ID
		s.non_overlapped_group_sets_IDs_pointer_host[i] = s.length;
		for (int j = 1; j < group_sets_ID_range; j++)
		{ // j is another nonempty group_set ID
			if ((i & j) == 0)
			{ // i and j are non-overlapping group sets
				/* The & (bitwise AND) in C or C++ takes two numbers as operands and does AND on every bit of two numbers. The result of AND for each bit is 1 only if both bits are 1.
				https://www.programiz.com/cpp-programming/bitwise-operators */
				s.non_overlapped_group_sets_IDs.push_back(j);
				s.length++;
			}
		}
	}
	s.non_overlapped_group_sets_IDs_pointer_host[group_sets_ID_range + 1] = s.length;
	return s;
}
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
void exp_CPU_Hop(string path, string data_name, int T, int D, int task_start_num, int task_end_num)
{
	std::cout << "start.. " << endl;

	std::vector<int> generated_group_vertices;
	std::unordered_set<int> generated_group_vertices_hash;
	graph_hash_of_mixed_weighted instance_graph, generated_group_graph;
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
	int group_sets_ID_range = (1<<T) - 1;
	non_overlapped_group_sets s = graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(group_sets_ID_range);

	/*iteration*/
	std::cout << "------------------------------------------------------------" << endl;
	int rounds = 0, cpu_cost, gpu_cost;

	/*output*/
	ofstream outputFile;
	outputFile.precision(8);
	outputFile.setf(ios::fixed);
	outputFile.setf(ios::showpoint);
	outputFile.open(path+"result/exp_CPU_Hop_" + data_name + "_T" + to_string(T) + "_"+"D"+to_string(D)+"_" + to_string(task_start_num) + "-" + to_string(task_end_num) + ".csv");

	outputFile << "task_ID,task,CPU_Hop_time,CPU_Hop_cost,CPU_Hop_memory,counts,process_num" << endl;
	int N = V,G = inquire[0].size();
	for (int i = task_start_num; i <= task_end_num; i++)
	//for (int i = task_end_num; i >= task_start_num; i--)
	{
		rounds++;

		std::cout << data_name << " iteration " << i << endl;

		string task = "";
		std::vector<int>().swap(generated_group_vertices);
		std::unordered_set<int>().swap(generated_group_vertices_hash);
		for (size_t j = 0; j < inquire[i].size(); j++)
		{
			generated_group_vertices.push_back(inquire[i][j]);
			generated_group_vertices_hash.insert(inquire[i][j]);
			
			task += to_string(inquire[i][j]) + " ";
		}
		std::cout << endl;

	
		if (1)
		{
				int RAM = 0;
				double time_record = 0;
				auto begin = std::chrono::high_resolution_clock::now();				
				records ret;
				graph_hash_of_mixed_weighted solu = HOP_cpu(generated_group_vertices_hash, v_generated_group_graph, v_instance_graph, D, time_record,RAM,ret);
				auto end = std::chrono::high_resolution_clock::now();				
				double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

				cpu_cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);
				cout<<"cost="<<cpu_cost<<endl;
				//graph_hash_of_mixed_weighted_print_size(solu);
				if (!this_is_a_feasible_solution_gpu(solu, v_generated_group_graph, generated_group_vertices))
				{
					std::cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
					graph_hash_of_mixed_weighted_print(solu);
					//  exit(1);
					cpu_cost= 100000;
				}
				std::cout << "------------------------------------------------------------" << endl;
				
				outputFile << i << "," << task << "," << time_record << "," << cpu_cost << "," << RAM<<"," <<ret.counts<<","<<ret.process_queue_num<< endl;

			
		}
	}
 

	outputFile << endl;
}
