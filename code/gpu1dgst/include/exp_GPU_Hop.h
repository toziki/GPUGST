#pragma once

#include <chrono>
#include <queue>
#include <omp.h>
#include <boost/heap/fibonacci_heap.hpp>
#include <graph_hash_of_mixed_weighted/graph_hash_of_mixed_weighted.h>
#include <graph_hash_of_mixed_weighted/common_algorithms/graph_hash_of_mixed_weighted_connected_components.h>
#include<DPQ.cuh>
#include <graph_hash_of_mixed_weighted_read_for_GSTP.h>
#include "graph_hash_of_mixed_weighted_sum_of_nw_ec.h"
#include <future>
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

void exp_GPU_Hop(string path, string data_name, int T, int D, int task_start_num, int task_end_num)
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
	CSR_graph csr_graph = toCSR(v_instance_graph);
	std::cout << "E: " << csr_graph.E_all << " V: " << csr_graph.V << endl;
	read_inquire(path + data_name + to_string(T) + ".csv", inquire);
	std::vector<int> cpu_costs(100);
	std::vector<double> cpu_times(100);
	int iteration_times = inquire.size();
	std::cout << "inquires size " << inquire.size() << " G = " << inquire[0].size() << endl;
	int group_sets_ID_range = pow(2, T) - 1;
	non_overlapped_group_sets s = graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(group_sets_ID_range);

	/*iteration*/
	std::cout << "------------------------------------------------------------" << endl;
	int rounds = 0, cpu_cost, gpu_cost;

	/*output*/
	ofstream outputFile;
	outputFile.precision(8);
	outputFile.setf(ios::fixed);
	outputFile.setf(ios::showpoint);
	outputFile.open(path+"result/exp_GPU1_Hop_" + data_name + "_T" + to_string(T) + "_" + "D" + to_string(D) + "_" + to_string(task_start_num) + "-" + to_string(task_end_num) + ".csv");

	outputFile << "task_ID,task,GPU1_Hop_time,GPU1_Hop_cost,GPU1_Hop_memory,counts,process_num" << endl;

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
		std::cout << endl;
		if (1)
		{

			int cost;
			int RAM;
			auto begin = std::chrono::high_resolution_clock::now();
			double runningtime;
			records ret;
			graph_hash_of_mixed_weighted solu = DPBF_gpu(csr_graph, generated_group_vertices, v_generated_group_graph, v_instance_graph, D, &runningtime, cost,RAM,ret);
			auto end = std::chrono::high_resolution_clock::now();
			//	runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
			// cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);
			outputFile << i << "," << task << "," << runningtime << "," << cost << "," <<RAM<<","<<ret.counts<<","<<ret.process_queue_num << endl;

			std::cout << "GPU cost: " << cost<< endl;
			// if (!this_is_a_feasible_solution_gpu(solu, v_generated_group_graph, generated_group_vertices))
			// {
			// 	std::cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
			// 	graph_hash_of_mixed_weighted_print_size(solu);
			// }
			std::cout << "------------------------------------------------------------" << endl;
		}
	}
			cudaFree(csr_graph.all_edge);
		cudaFree(csr_graph.all_pointer);
		cudaFree(csr_graph.all_edge_weight);
	outputFile << endl;
}
