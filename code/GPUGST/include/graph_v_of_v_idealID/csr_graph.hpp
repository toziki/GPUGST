#pragma once
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <queue>
#include <vector>
#include "graph_v_of_v_idealID.h"
#include <unordered_map>
/*for GPU*/
using namespace std;
inline bool compare_weight(std::pair<int, double> a, std::pair<int, double> b)
{
    return a.second < b.second; // 升序排列
}
class CSR_graph
{
public:
    std::vector<int> INs_Neighbor_start_pointers, OUTs_Neighbor_start_pointers, ALL_start_pointers; // degree_vector Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights
    /*
        Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
        And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    */
    std::vector<int> INs_Edges, OUTs_Edges, all_Edges;                      // Edges[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] neighbor IDs
    std::vector<int> INs_Edge_weights, OUTs_Edge_weights, ALL_Edge_weights; // Edge_weights[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] edge weights
    std::vector<int> v_pointer, map_virtual_To_real, v_edges, v_weight;
    std::vector<std::pair<int, double>> edges;
    int *in_pointer, *out_pointer, *in_edge, *out_edge, *all_pointer, *all_edge, *degrees;
    int *in_edge_weight, *out_edge_weight, *all_edge_weight, *virtual_to_real;
    int E_all = 0, V, max_d = 0,ave_d,ave_w;
    std::vector<int> pto;
    std::unordered_map<int, int> otn;
};

// CSR_graph<weight_type> toCSR(graph_structure<weight_type>& graph)
// inline graph_v_of_v_idealID tovir(graph_v_of_v_idealID graph)
// {

// }

inline CSR_graph toCSR(graph_v_of_v_idealID graph)
{
    // cudaSetDevice(1);
    CSR_graph ARRAY;
    int V = graph.size();
    ARRAY.V = V;
    // ARRAY.INs_Neighbor_start_pointers.resize(V + 1); // Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    // ARRAY.OUTs_Neighbor_start_pointers.resize(V + 1);
    ARRAY.ALL_start_pointers.resize(V + 1);
    // ARRAY.degree_vector.resize(V + 1);
    int pointer = 0;
    int maxd = 0;

    for (int i = 0; i < V; i++)
    {
        ARRAY.ALL_start_pointers[i] = pointer;
        // ARRAY.degree_vector[i] = graph[i].size();
        for (int j = 0; j < graph[i].size(); j++)
        {
            ARRAY.edges.push_back(graph[i][j]);
            ARRAY.E_all++;
        }
      //  std::sort(ARRAY.edges.begin() + pointer, ARRAY.edges.begin() + pointer + graph[i].size(), compare_weight);
        maxd = max(int(graph[i].size()), maxd);
        ARRAY.ave_d+=graph[i].size();
        pointer += graph[i].size();
    }
        ARRAY.ALL_start_pointers[V] = pointer;
    // for (size_t i = 0; i < V; i++)
    // {
    //     cout<<i<<" ";
    //     for (size_t j = ARRAY.ALL_start_pointers[i]; j <  ARRAY.ALL_start_pointers[i+1]; j++)
    //     {
    //       cout<<ARRAY.edges[i].first<<"-"<<ARRAY.edges[i].second<<" ";
    //     }
    //     cout<<endl;
    // }
    
    for (int i = 0; i < V; i++)
    {
        for (int j = ARRAY.ALL_start_pointers[i]; j < ARRAY.ALL_start_pointers[i + 1]; j++)
        {
            ARRAY.all_Edges.push_back(ARRAY.edges[j].first);
            ARRAY.ALL_Edge_weights.push_back(ARRAY.edges[j].second);
             ARRAY.ave_w+=ARRAY.edges[j].second;
            
        }
    }

    ARRAY.max_d = maxd;
   

    // for (int i = 0; i < V; i++)
    // {
    //     if (graph[i].size()>K)
    //     {
    //         queue<std::pair<int, int>>q;
    //         for (size_t j = ARRAY.ALL_start_pointers[i]; j < ARRAY.ALL_start_pointers[i+1]; j++)
    //         {
    //             q.push({ARRAY.all_Edges[j],ARRAY.all_Edges_weight[j]});
    //         }
    //         int n=0;
    //         while (q.size()>K)
    //         {

    //             for (int j = 0; j < K; j++)
    //         {

    //         }
    //         }

    //     }

    // }

    int E_all = ARRAY.E_all;
     ARRAY.ave_d/=V,ARRAY.ave_w/=E_all;
      cout << "max degree " << maxd<<" aved and avew "<<ARRAY.ave_d<<" " <<ARRAY.ave_w<< endl;
    cudaMallocManaged(&ARRAY.all_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged(&ARRAY.all_edge, E_all * sizeof(int));
    cudaMallocManaged(&ARRAY.all_edge_weight, E_all * sizeof(int));
    // cudaMallocManaged(&ARRAY.degrees, (V + 1) * sizeof(int));

    // cudaMemcpy(ARRAY.degrees, ARRAY.ALL_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.all_pointer, ARRAY.ALL_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.all_edge, ARRAY.all_Edges.data(), E_all * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.all_edge_weight, ARRAY.ALL_Edge_weights.data(), E_all * sizeof(int), cudaMemcpyHostToDevice);

    return ARRAY;
}

//   for (int i = 0; i < V; i++)
//     {
//         ARRAY.ALL_start_pointers[i] = pointer;
//         for (int j = 0; j < graph[i].size(); j++)
//         {
//             ARRAY.all_Edges.push_back(graph[i][j].first);
//             ARRAY.ALL_Edge_weights.push_back(graph[i][j].second);
//             ARRAY.E_all++;
//         }

//         pointer += graph[i].size();
//     }
//     ARRAY.ALL_start_pointers[V] = pointer;