

using namespace std;

struct queue_element_d
{
    int v, p, d, cost;

    queue_element_d(int _v = 0, int _p = 0, int _d = 0, int _cost = 0)
        : v(_v), p(_p), d(_d), cost(_cost) {}

    bool operator>(const queue_element_d &k) const
    {
        return cost > k.cost;
    }

};
struct records
{   int process_queue_num;
    int counts;
};
#define inf 100000
struct node
{
	int update = 0;
	int type;				 // =0: this is the single vertex v; =1: this tree is built by grown; =2: built by merge
	int cost = inf, lb, lb1; // cost of this tree T(v,p);
	int u;					 // if this tree is built by grow, then it's built by growing edge (v,u,d-1);
	int p1, p2, d1, d2;		 // if this tree is built by merge, then it's built by merge T(v,p1,d1) and T(v,p2,d2);
};
bool operator<(  queue_element_d const &x,   queue_element_d const &y)
{
	return x.cost > y.cost; // < is the max-heap; > is the mean heap; PriorityQueue is expected to be a max-heap of integer values
}
typedef typename boost::heap::fibonacci_heap<  queue_element_d>::handle_type handle_graph_v_of_v_idealID_PrunedDPPlusPlus_min_node;
void set_max_ID(graph_v_of_v_idealID &group_graph, std::unordered_set<int> &cumpulsory_group_vertices, vector<vector<vector<node>>>& host_tree, std::unordered_set<int> &contain_group_vertices)
{
    int bit_num = 1, v;
    for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++, bit_num <<= 1)
    {
        for (size_t to = 0; to < group_graph[*it].size(); to++)
        {
            v = group_graph[*it][to].first;
            host_tree[v][bit_num][0].cost = 0;
            contain_group_vertices.insert(v);
        }
    }
}
int get_max(int vertex, vector<vector<vector<node>>>& host_tree, int width)
{
    int re = 0;
    for (size_t i = 1; i < width; i <<= 1)
    {
        if (host_tree[vertex][i][0].cost == 0)
        {
            re += i;
        }
    }

    return re;
}
vector<vector<int>> non_overlapped_group;

inline int graph_v_of_v_idealID_DPBF_vertex_group_set_ID_gpu(int vertex, graph_v_of_v_idealID &group_graph,
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
            ID = ID + (1 << pow_num);
        }
        pow_num++;
    }

    return ID;
}

void graph_v_of_v_idealID_DPBF_non_overlapped_group(int group_sets_ID_range)
{

    /*this function calculate the non-empty and non_overlapped_group_sets_IDs of each non-empty group_set ID;
    time complexity: O(4^|Gamma|), since group_sets_ID_range=2^|Gamma|;
    the original DPBF code use the same method in this function, and thus has the same O(4^|Gamma|) complexity;*/
    vector<vector<int>>().swap(non_overlapped_group);
    non_overlapped_group.resize(group_sets_ID_range + 2);
    for (int i = 1; i <= group_sets_ID_range; i++)
    { // i is a nonempty group_set ID
        for (int j = 1; j <= group_sets_ID_range; j++)
        { // j is another nonempty group_set ID
            if ((i & j) == 0)
            {
                non_overlapped_group[i].push_back(j);
                // i and j are non-overlapping group sets
                /* The & (bitwise AND) in C or C++ takes two numbers as operands and does AND on every bit of two numbers. The result of AND for each bit is 1 only if both bits are 1.
                https://www.programiz.com/cpp-programming/bitwise-operators */
            }
        }
    }
}

graph_hash_of_mixed_weighted HOP_cpu(std::unordered_set<int> &cumpulsory_group_vertices, graph_v_of_v_idealID &group_graph, graph_v_of_v_idealID &input_graph, int D, double &time_record,int &RAM,records &ret)
{
    int N = input_graph.size();
    int G = cumpulsory_group_vertices.size();
   
    int group_sets_ID_range = (1 << G) - 1;
    // node ***host_tree = new node **[N];
    // node **host_tree_two = new node *[N * (1 << G)];
    // node *host_tree_one = new node[N * (1 << G) * (D + 3)];
    // for (int i = 0; i < N; i++)
    // {
    //     host_tree[i] = host_tree_two + i * (1 << G); // 连续的行
    //     for (int j = 0; j < (1 << G); j++)
    //     {
    //         host_tree[i][j] = host_tree_one + (i * (1 << G) + j) * (D + 3); // 偏移
    //     }
    // }


    vector<vector<vector<node>>> host_tree(N);
    for (int i = 0; i < N; i++)
    {
        host_tree[i].resize(1 << G);
        for (int j = 0; j < (1 << G); j++)
        {
            host_tree[i][j].resize(D + 1); 
        }
    }


    // node host_tree[N][1 << G][D + 3];
    priority_queue<queue_element_d, vector<queue_element_d>, greater<queue_element_d>> q;
 //  	boost::heap::fibonacci_heap<queue_element_d> q;
    queue<queue_element_d> Q;
    graph_hash_of_mixed_weighted solution_tree;

    graph_v_of_v_idealID_DPBF_non_overlapped_group(group_sets_ID_range);
    for (int v = 0; v < N; v++)
    {
        for (int p = 0; p <= group_sets_ID_range; p++)
        {
            for (int d = 0; d <= D; d++)
            {
                host_tree[v][p][d].cost = inf;
            }
        }
    }

    std::unordered_set<int> contain_group_vertices;
    int width = 1 << G;
    set_max_ID(group_graph, cumpulsory_group_vertices, host_tree, contain_group_vertices);
    for (int v = 0; v < N; v++)
    {
        int group_set_ID_v = get_max(v, host_tree, width);
        for (int i = 1; i <= group_set_ID_v; i <<= 1)
        {
            if (i & group_set_ID_v)
            {
                host_tree[v][i][0].cost = 0;
                host_tree[v][i][0].type = 0;
                q.push(queue_element_d(v, i, 0, 0));
            }
        }
    }
    auto begin = std::chrono::high_resolution_clock::now();
    int max_queue_size = 0;
    int process =0;
    while (q.size())
    {
        process++;
        max_queue_size = max(max_queue_size,int(q.size()));
        queue_element_d top_node = q.top();
        q.pop();
        int v = top_node.v;
        int p = top_node.p;
        int cost = top_node.cost;
        int d = top_node.d;
        if (cost != host_tree[v][p][d].cost)
            continue;
        if (p == group_sets_ID_range)
        {
            Q.push(queue_element_d(v, p, d));
            break;
        }
        // grow
        if (d < D)
        {
            for (auto edge : input_graph[v])
            {
                int u = edge.first;
                int len = edge.second;
                if (host_tree[u][p][d + 1].cost > cost + len)
                {
                    host_tree[u][p][d + 1].cost = cost + len;
                    host_tree[u][p][d + 1].type = 1;
                    host_tree[u][p][d + 1].u = v;
                    q.push(queue_element_d(u, p, d + 1, cost + len));
                }
            }
        }
        // merge
        int p1 = p, d1 = d;
        for (auto p2 : non_overlapped_group[p1])
        {
            for (int d2 = 0; d2 <= D - d1; d2++)
            {
                int p1_cup_p2 = p1 | p2;
                int new_d = max(d1, d2);
                int merge_tree_cost = cost + host_tree[v][p2][d2].cost;
                if (host_tree[v][p1_cup_p2][new_d].cost > merge_tree_cost)
                {
                    host_tree[v][p1_cup_p2][new_d].cost = merge_tree_cost;
                    host_tree[v][p1_cup_p2][new_d].type = 2;
                    host_tree[v][p1_cup_p2][new_d].p1 = p1;
                    host_tree[v][p1_cup_p2][new_d].p2 = p2;
                    host_tree[v][p1_cup_p2][new_d].d1 = d1;
                    host_tree[v][p1_cup_p2][new_d].d2 = d2;
                    q.push(queue_element_d(v, p1_cup_p2, new_d, merge_tree_cost));
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    time_record = runningtime;
 
    begin = std::chrono::high_resolution_clock::now();
    while (Q.size())
    {
        queue_element_d statu = Q.front();
        Q.pop();
        int v = statu.v;
        int p = statu.p;
        int d = statu.d;
        // graph_hash_of_mixed_weighted_add_vertex(graph_hash_of_mixed_weighted& input_graph, int vertex, double weight) {
        graph_hash_of_mixed_weighted_add_vertex(solution_tree, v, 0);
        if (host_tree[v][p][d].type == 1)
        {
            int u = host_tree[v][p][d].u;
            int c_uv = graph_v_of_v_idealID_edge_weight(input_graph, u, v);
            graph_hash_of_mixed_weighted_add_edge(solution_tree, u, v, c_uv);
            Q.push(queue_element_d(u, p, d - 1));
        }
        if (host_tree[v][p][d].type == 2)
        {
            int p1 = host_tree[v][p][d].p1;
            int p2 = host_tree[v][p][d].p2;
            int d1 = host_tree[v][p][d].d1;
            int d2 = host_tree[v][p][d].d2;
            Q.push(queue_element_d(v, p1, d1));
            Q.push(queue_element_d(v, p2, d2));
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
 runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    // delete []host_tree;
    // delete []host_tree_one;
    // delete []host_tree_two;
    int counts = 0;
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            for (size_t d = 0; d <= D; d++)
            {
                if (host_tree[i][j][d].cost!=inf)
                {
                    counts++;
                }
                
            }
            
        }
        
    }
       ret.counts =  counts;
       ret.process_queue_num = process;
    //cout<<"queue_size "<<max_queue_size<<"N*width*D "<<N*width*D<<"count "<<counts<<endl;
    RAM = ((max_queue_size+N*width*D+counts));
    return solution_tree;
}
