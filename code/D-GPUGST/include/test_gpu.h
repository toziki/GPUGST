#pragma once

#include <chrono>
#include <queue>
#include <omp.h>
#include <boost/heap/fibonacci_heap.hpp>
#include <graph_hash_of_mixed_weighted/graph_hash_of_mixed_weighted.h>
#include <graph_hash_of_mixed_weighted/common_algorithms/graph_hash_of_mixed_weighted_connected_components.h>
#include <graph_hash_of_mixed_weighted/random_graph/graph_hash_of_mixed_weighted_generate_random_connected_graph.h>
#include <graph_hash_of_mixed_weighted/two_graphs_operations/graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID.h>
#include <graph_hash_of_mixed_weighted_generate_random_groups_of_vertices.h>
#include <graph_hash_of_mixed_weighted_save_for_GSTP.h>
#include <graph_hash_of_mixed_weighted_read_for_GSTP.h>
#include "graph_hash_of_mixed_weighted_sum_of_nw_ec.h"
#include "graph_v_of_v_idealID_DPBF_only_ec.h"
#include "graph_v_of_v_idealID_PrunedDPPlusPlus.h"
#include "DPQ.cuh"
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

void test_graph_v_of_v_idealID_DPBF_only_ec_gpu()
{
	std::cout << "start.. " << endl;
	/*parameters*/
	int iteration_times = 1;
	int V = 300000, E = 800000, G = 6, g_size_min = 1, g_size_max = 4, precision = 0;
	int ec_min = 1, ec_max = 4; // PrunedDP does not allow zero edge weight

	int solution_cost_GPU_sum = 0, solution_cost_PrunedDPPlusPlus_sum = 0, solution_cost_multi_GPU_sum = 0;
	double time_GPU_one_avg = 0, time_PrunedDPPlusPlus_avg = 0, time_GPU_multi_avg = 0;
	int p_gpu = 0, p_cpu = 0;
	int *pointer1 = &p_gpu, *pointer2 = &p_cpu;
	int generate_new_graph = 0;
	int lambda = 1;
	std::vector<int> generated_group_vertices;
	std::unordered_set<int> generated_group_vertices_hash;
	graph_hash_of_mixed_weighted instance_graph, generated_group_graph;
	std::vector<std::vector<int>> inquire;
	graph_v_of_v_idealID v_generated_group_graph, v_instance_graph;
	/*if (generate_new_graph == 1)
	{
		instance_graph = graph_hash_of_mixed_weighted_generate_random_connected_graph(V, E, 0, 0, ec_min, ec_max, precision);

		graph_hash_of_mixed_weighted_generate_random_groups_of_vertices(G, g_size_min, g_size_max,
																		instance_graph, instance_graph.hash_of_vectors.size(), generated_group_vertices, generated_group_graph); //

		graph_hash_of_mixed_weighted_save_for_GSTP("simple_iterative_tests.text", instance_graph,
												   generated_group_graph, generated_group_vertices, lambda);
		graph_hash_of_mixed_weighted_read_for_GSTP_1(data_name+"_in", instance_graph,
												 generated_group_graph, generated_group_vertices, lambda, community);
	cout << "community size " << community.size() << endl;
	graph_hash_of_mixed_weighted_read_for_Group(data_name+".g", instance_graph,
												generated_group_graph, generated_group_vertices);
	cout << "Group graph size " << generated_group_graph.hash_of_vectors.size() << endl;
	graph_hash_of_mixed_weighted_read_for_inquire(data_name+".g5", inquire);
	}*///         直径约束cpu和gpu
	std::vector<std::string> datas = {"dblp"};
	//,"youtu","orkut","musae","Github"
	std::vector<std::string> tails = {"3","5","7"};

		string data_name = datas[ii];
		int ov = read_input_graph(data_name + ".in", v_instance_graph);
		std::cout << "read input complete" << endl;
		read_Group(data_name + ".g", v_instance_graph, v_generated_group_graph);

		std::cout << "read group complete " << v_generated_group_graph.size() << endl;
		std::cout << "enter " << endl;

		V = v_instance_graph.size();
		CSR_graph csr_graph = toCSR(v_instance_graph);
		std::cout << "E: " << csr_graph.E_all << " V: " << csr_graph.V << endl;
			for (int i = 0; i < 2; i++)
			{
				cout << i << "  ";
				for (int j = csr_graph.all_pointer[i]; j < csr_graph.all_pointer[i + 1]; j++)
				{
					std::cout << csr_graph.all_edge[j] << "-" << csr_graph.all_edge_weight[j] << " ";
				}
				std::cout << endl;
			}
		for (int jj = 0; jj < tails.size(); jj++)
		{
			inquire.clear();
			read_inquire(data_name + ".g" + tails[jj], inquire);
			std::vector<int> gpu_costs(100), cpu_costs(100);
			std::vector<double> gpu_times(100), cpu_times(100);
			
			int solve = 0;
			iteration_times = inquire.size();
			std::cout << "inquires size " << inquire.size() << " G = " << inquire[0].size() << endl;

		
			G = inquire[0].size();
			std::cout << "gsize " << G << endl;
			int group_sets_ID_range = pow(2, G) - 1;
			non_overlapped_group_sets s = graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(group_sets_ID_range);
			/*iteration*/
			std::cout << "------------------------------------------------------------" << endl;
			int rounds = 0;
			solution_cost_PrunedDPPlusPlus_sum = 0;solution_cost_GPU_sum=0;time_PrunedDPPlusPlus_avg=0;
			time_GPU_one_avg = 0;
			for (int i = 0; i < 100; i++)
			{
				rounds++;

				std::cout << "iteration " << i << endl;

				// generated_group_graph.clear();
				generated_group_vertices.clear();
				generated_group_vertices_hash.clear();
				for (size_t j = 0; j < inquire[i].size(); j++)
				{
					generated_group_vertices.push_back(inquire[i][j]);
					generated_group_vertices_hash.insert(inquire[i][j]);
					// std::cout << v_generated_group_graph[inquire[i][j]-ov].size() << " ";
				}
				std::cout << endl;
				/*graph_hash_of_mixed_weighted_generate_community_groups_of_vertices(G, g_size_min, g_size_max,
																				   instance_graph, instance_graph.hash_of_vectors.size(), generated_group_vertices, generated_group_graph, 20, belong, community); //
		*/

				std::cout << "get inquire complete" << std::endl;
				if (1)
				{
					int RAM;
					auto begin = std::chrono::high_resolution_clock::now();
					double *time_record = &cpu_times[i];
					graph_hash_of_mixed_weighted solu = HOP_cpu(generated_group_vertices_hash, v_generated_group_graph, v_instance_graph, 5, time_record);
					auto end = std::chrono::high_resolution_clock::now();
					double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
					time_PrunedDPPlusPlus_avg += *time_record;

					// graph_hash_of_mixed_weighted_print(solu);

					int cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);
					cpu_costs[i] = cost;
					if (cpu_costs[i]!=inf)
					{
						
						solution_cost_PrunedDPPlusPlus_sum = solution_cost_PrunedDPPlusPlus_sum + cost;
					}
					
					
					std::cout << "CPU weight " << cpu_costs[i] << " time " << cpu_times[i] << endl;
					// graph_hash_of_mixed_weighted_print_size(solu);
					if (!this_is_a_feasible_solution_gpu(solu, v_generated_group_graph, generated_group_vertices))
					{
						std::cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
						// graph_hash_of_mixed_weighted_print(solu);
						//   exit(1);
					}
					std::cout << "------------------------------------------------------------" << endl;
				}

				
			write_result_cpu(data_name, cpu_times, cpu_costs, tails[jj]);
			cout<< "solve "<<solve<<endl;
			std::cout << "solution_cost_CPU  _sum=" << solution_cost_PrunedDPPlusPlus_sum << endl;
			std::cout << "time_CPU avg=" << time_PrunedDPPlusPlus_avg / 100 << "s" << endl;
		}
		cudaFree(csr_graph.all_edge);
		cudaFree(csr_graph.all_pointer);
		cudaFree(csr_graph.all_edge_weight);
	}

	 datas[0] = {"orkut"};
	//,"youtu","orkut","musae","Github"
	tails[0] = "3";
	tails.push_back("5");
	// for (int ii = 0; ii < datas.size(); ii++)
	// {
	// 	string data_name = datas[ii];
	// 	int ov = read_input_graph(data_name + ".in", v_instance_graph);
	// 	std::cout << "read input complete" << endl;
	// 	read_Group(data_name + ".g", v_instance_graph, v_generated_group_graph);

	// 	std::cout << "read group complete " << v_generated_group_graph.size() << endl;
	// 	std::cout << "enter " << endl;

	// 	V = v_instance_graph.size();
	// 	CSR_graph csr_graph = toCSR(v_instance_graph);
	// 	std::cout << "E: " << csr_graph.E_all << " V: " << csr_graph.V << endl;
	// 		for (int i = 0; i < 2; i++)
	// 		{
	// 			cout << i << "  ";
	// 			for (int j = csr_graph.all_pointer[i]; j < csr_graph.all_pointer[i + 1]; j++)
	// 			{
	// 				std::cout << csr_graph.all_edge[j] << "-" << csr_graph.all_edge_weight[j] << " ";
	// 			}
	// 			std::cout << endl;
	// 		}
	// 	for (int jj = 0; jj < tails.size(); jj++)
	// 	{
	// 		inquire.clear();
	// 		read_inquire(data_name + ".g" + tails[jj], inquire);
	// 		std::vector<int> gpu_costs(100), cpu_costs(100);
	// 		std::vector<double> gpu_times(100), cpu_times(100);
			
	// 		int solve = 0;
	// 		iteration_times = inquire.size();
	// 		std::cout << "inquires size " << inquire.size() << " G = " << inquire[0].size() << endl;

		
	// 		G = inquire[0].size();
	// 		std::cout << "gsize " << G << endl;
	// 		int group_sets_ID_range = pow(2, G) - 1;
	// 		non_overlapped_group_sets s = graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(group_sets_ID_range);
	// 		/*iteration*/
	// 		std::cout << "------------------------------------------------------------" << endl;
	// 		int rounds = 0;
	// 		solution_cost_PrunedDPPlusPlus_sum = 0;solution_cost_GPU_sum=0;time_PrunedDPPlusPlus_avg=0;
	// 		time_GPU_one_avg = 0;
	// 		for (int i = 0; i < 100; i++)
	// 		{
	// 			rounds++;

	// 			std::cout << "iteration " << i << endl;

	// 			// generated_group_graph.clear();
	// 			generated_group_vertices.clear();
	// 			generated_group_vertices_hash.clear();
	// 			for (size_t j = 0; j < inquire[i].size(); j++)
	// 			{
	// 				generated_group_vertices.push_back(inquire[i][j]);
	// 				generated_group_vertices_hash.insert(inquire[i][j]);
	// 				// std::cout << v_generated_group_graph[inquire[i][j]-ov].size() << " ";
	// 			}
	// 			std::cout << endl;
	// 			/*graph_hash_of_mixed_weighted_generate_community_groups_of_vertices(G, g_size_min, g_size_max,
	// 																			   instance_graph, instance_graph.hash_of_vectors.size(), generated_group_vertices, generated_group_graph, 20, belong, community); //
	// 	*/

	// 			std::cout << "get inquire complete" << std::endl;
	// 			if (1)
	// 			{
	// 				int RAM;
	// 				auto begin = std::chrono::high_resolution_clock::now();
	// 				double *time_record = &cpu_times[i];
	// 				graph_hash_of_mixed_weighted solu = DPBF_cpu(generated_group_vertices_hash, v_generated_group_graph, v_instance_graph, 5, time_record);
	// 				auto end = std::chrono::high_resolution_clock::now();
	// 				double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	// 				time_PrunedDPPlusPlus_avg += *time_record;

	// 				// graph_hash_of_mixed_weighted_print(solu);

	// 				int cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);
	// 				cpu_costs[i] = cost;
	// 				if (cpu_costs[i]!=inf)
	// 				{
						
	// 					solution_cost_PrunedDPPlusPlus_sum = solution_cost_PrunedDPPlusPlus_sum + cost;
	// 				}
					
					
	// 				std::cout << "CPU weight " << cpu_costs[i] << " time " << cpu_times[i] << endl;
	// 				// graph_hash_of_mixed_weighted_print_size(solu);
	// 				if (!this_is_a_feasible_solution_gpu(solu, v_generated_group_graph, generated_group_vertices))
	// 				{
	// 					std::cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
	// 					// graph_hash_of_mixed_weighted_print(solu);
	// 					//   exit(1);
	// 				}
	// 				std::cout << "------------------------------------------------------------" << endl;
	// 			}

	// 			if (1)
	// 			{

	// 				int cost;
	// 				int RAM, *real_cost = &cost;
	// 				auto begin = std::chrono::high_resolution_clock::now();
	// 				double runningtime;
	// 				graph_hash_of_mixed_weighted solu = DPBF_gpu(csr_graph, generated_group_vertices, v_generated_group_graph, v_instance_graph, 5, &runningtime, real_cost);
	// 				auto end = std::chrono::high_resolution_clock::now();
	// 				gpu_times[i] = runningtime;
	// 				gpu_costs[i] = *real_cost;
	// 				//	runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
	// 				time_GPU_one_avg += runningtime;

	// 				// cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);
	// 				if (gpu_costs[i]!=inf)
	// 				{	solve++;
	// 					solution_cost_GPU_sum += *real_cost;
	// 				}
					
					
	// 				std::cout << "GPU cost: " << *real_cost << " time " << runningtime << endl;
	// 				// if (!this_is_a_feasible_solution_gpu(solu, v_generated_group_graph, generated_group_vertices))
	// 				// {
	// 				// 	std::cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
	// 				// 	graph_hash_of_mixed_weighted_print_size(solu);
	// 				// }
	// 				std::cout << "------------------------------------------------------------" << endl;
	// 			}
	// 		}
	// 		write_result_cpu(data_name, cpu_times, cpu_costs, tails[jj]);
	// 		write_result_gpu(data_name, gpu_times, gpu_costs, tails[jj]);
	// 		cout<< "solve "<<solve<<endl;
	// 		std::cout << "solution_cost_CPU  _sum=" << solution_cost_PrunedDPPlusPlus_sum << endl;
	// 		std::cout << "solution_cost_GPU_1_sum=" << solution_cost_GPU_sum << endl;
	// 		std::cout << "time_CPU avg=" << time_PrunedDPPlusPlus_avg / solve << "s" << endl;
	// 		std::cout << "time_GPU avg=" << time_GPU_one_avg / solve << "s" << endl;
	// 	}
	// 	cudaFree(csr_graph.all_edge);
	// 	cudaFree(csr_graph.all_pointer);
	// 	cudaFree(csr_graph.all_edge_weight);
	// }
}
