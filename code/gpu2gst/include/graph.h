#ifndef __GRAPH_H__
#define __GRAPH_H__
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "wtime.h"
#include<vector>
#include<bits/stdc++.h>
#include<read_group.hpp>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
typedef std::vector<std::vector<std::pair<int,int>>> graph_v_of_v_idealID;
typedef struct non_overlapped_group_sets{
	std::vector<int> non_overlapped_group_sets_IDs_pointer_host, non_overlapped_group_sets_IDs;
	int length;
}non_overlapped_group_sets;
void set_max_ID(graph_v_of_v_idealID &group_graph, std::vector<int> &cumpulsory_group_vertices, int *host_tree, std::vector<int> &contain_group_vertices,int width)
{
	int bit_num = 1, v;
	for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++, bit_num <<= 1)
	{
		for (size_t to = 0; to < group_graph[*it].size(); to++)
		{
			v = group_graph[*it][to].first;
			host_tree[v*width+bit_num] = 0;
			contain_group_vertices.push_back(v);
		}
		
	}
	
}
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
inline off_t fsize(const char *filename) {
	struct stat st; 
	if (stat(filename, &st) == 0)
		return st.st_size;
	return -1; 
}

template<
typename file_vert_t, typename file_index_t, typename file_weight_t,
typename new_vert_t, typename new_index_t, typename new_weight_t>
class graph
{
	public:
		new_index_t *beg_pos;
		new_vert_t *adj_list;
		new_weight_t *weight;
		new_index_t vert_count;
		new_index_t edge_count;
		graph_v_of_v_idealID group_graph;
		std::vector<std::vector<int>>inquire;
	public:
		graph(){};
		~graph(){};
		graph(const char *beg_file, 
				const char *adj_list_file,
				const char *weight_file);

		graph(file_vert_t *csr,
				file_index_t *beg_pos,
				file_weight_t *weight_list,
				file_index_t vert_count,
				file_index_t edge_count,
				graph_v_of_v_idealID group_graph,
		std::vector<std::vector<int>>inquire
				)
		{
			this->beg_pos = beg_pos;
			this->adj_list = csr;
			this->weight = weight_list;
			this->edge_count = edge_count;
			this->vert_count = vert_count;
			this->group_graph = group_graph;
			this->inquire = inquire;
		};

		void read_gi(std::string file_name1,std::string file_name2)
		{
			read_Group(file_name1,this->group_graph);
			std::cout<<"read group over"<<std::endl;
			read_inquire(file_name2,this->inquire);
			std::cout<<"read inquire over"<<std::endl;
			//non_overlapped_group_sets((1<<(this->inquire[0].size())-1));
		};
};
int get_max(int vertex, int *host_tree, int width)
{
	int re = 0;
	for (size_t i = 1; i < width; i <<= 1)
	{
		if (host_tree[vertex*width+i] == 0)
		{
			re += i;
		}
	}

	return re;
}



#include <graph.hpp>
#endif
