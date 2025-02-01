#pragma once

#include <text_mining/parse_string.h>
bool compare(std::pair<int, int> a, std::pair<int, int> b)
{
	return a.first < b.first; // 升序排列
}







int read_input_graph(std::string instance_name, graph_v_of_v_idealID &input_graph)

{



	std::string line_content;
	std::ifstream myfile(instance_name); // open the file
	if (myfile.is_open()) // if the file is opened successfully
	{

	getline(myfile, line_content);
	input_graph.clear();
	std::vector<std::string> Parsed_content = parse_string(line_content, " ");
	int V = std::stod(Parsed_content[0]);
	cout << "V " << Parsed_content[0] << endl;
	input_graph.resize(V);
	int K = 64;

		while (getline(myfile, line_content)) // read file line by line
		{
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");
		if (Parsed_content.size()==3)
			{
				int v1 = std::stoi(Parsed_content[0]);
				int v2 = std::stoi(Parsed_content[1]);
				int ec = std::stoi(Parsed_content[2]);
				input_graph[v1].push_back({v2, ec});
				input_graph[v2].push_back({v1, ec});
			}
		}


		
		for (size_t i = 0; i < input_graph.size(); i++)
		{
			std::sort(input_graph[i].begin(), input_graph[i].end());
		}

		myfile.close(); // close the file

		return V;
	}
	else
	{
		std::cout << "Unable to open file " << instance_name << std::endl
				  << "Please check the file location or file name." << std::endl; // throw an error message
		getchar();																  // keep the console window
		exit(1);																  // end the program
	}
	
}




void graph_hash_of_mixed_weighted_read_for_Group(std::string instance_name,
												 graph_hash_of_mixed_weighted &input_graph, graph_hash_of_mixed_weighted &group_graph,
												 std::unordered_set<int> &group_vertices)
{

	std::string line_content;
	std::ifstream myfile(instance_name); // open the file
	int vnum = input_graph.hash_of_vectors.size();
	for (size_t i = 0; i < vnum; i++)
	{
		graph_hash_of_mixed_weighted_add_vertex(group_graph, i, 0);
	}

	if (myfile.is_open()) // if the file is opened successfully
	{





		while (getline(myfile, line_content)) // read file line by line
		{
			std::vector<std::string> Parsed_content = parse_string(line_content, ":");
			int g = std::stod(Parsed_content[0].substr(1, Parsed_content[0].length())) + vnum;
			graph_hash_of_mixed_weighted_add_vertex(group_graph, g, 0);
			std::vector<std::string> groups = parse_string(Parsed_content[1], " ");
			for (size_t i = 0; i < groups.size(); i++)
			{
				int v = std::stoi(groups[i]);
				graph_hash_of_mixed_weighted_add_edge(group_graph, g, v, 1);
			}
		}

		myfile.close(); // close the file
	}
	else
	{
		std::cout << "Unable to open file " << instance_name << std::endl
				  << "Please check the file location or file name." << std::endl; // throw an error message
		getchar();																  // keep the console window
		exit(1);																  // end the program
	}
}


void read_Group(std::string instance_name, graph_v_of_v_idealID &input_graph, graph_v_of_v_idealID &group_graph)
{

	std::string line_content;
	std::ifstream myfile(instance_name); // open the file
	group_graph.clear();
	// group_graph.resize(input_graph.size());
	if (myfile.is_open()) // if the file is opened successfully
	{
		while (getline(myfile, line_content)) // read file line by line
		{
			if (line_content[line_content.length() - 1] == ' ')
			{
				line_content.erase(line_content.length() - 1);
			}
			std::vector<std::string> Parsed_content = parse_string(line_content, ":");
			int g = std::stod(Parsed_content[0].substr(1, Parsed_content[0].length())) - 1;
			group_graph.push_back({});


           
//cout << g << endl;

			//input_graph.push_back({});
			std::stringstream ss(Parsed_content[1]);
			std::istream_iterator<std::string> begin(ss);
			std::istream_iterator<std::string> end;
			std::vector<std::string> groups_mem(begin, end);
			group_graph[g].resize(groups_mem.size());
			//input_graph[input_graph.size()-1].resize(groups_mem.size());





			for (size_t i = 0; i < groups_mem.size(); i++)
			{
				int v = std::stoi(groups_mem[i]);
				group_graph[g][i] = {v, 1};
				//input_graph[input_graph.size()-1][i] = {v,0};
			}

			std::sort(group_graph[g].begin(), group_graph[g].end(), compare);
		}

		cout << endl;
		myfile.close(); // close the file
	}
	else
	{
		std::cout << "Unable to open file " << instance_name << std::endl
				  << "Please check the file location or file name." << std::endl; // throw an error message
		getchar();																  // keep the console window
		exit(1);																  // end the program
	}
}
void read_inquire(std::string instance_name, std::vector<std::vector<int>> &inquire)
{
	//cout << "open inquire " <<instance_name<<endl;
	std::string line_content;
	std::ifstream myfile(instance_name); // open the file
	inquire.clear();
	if (myfile.is_open())				 // if the file is opened successfully
	{
		cout << "success open inquire " <<instance_name<<endl;
		while (getline(myfile, line_content)) // read file line by line
		{
			inquire.push_back({});
			
			if (line_content[line_content.length() - 1] == ' ')
			{
				line_content.erase(line_content.length() - 1);
			}
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");
			
			for (size_t i = 0; i < Parsed_content.size(); i++)
			{
				
				int v = std::stoi(Parsed_content[i]);
				
				inquire[inquire.size() - 1].push_back(v);
			}
			// for (size_t i = 0; i < inquire.size(); i++)
			// {
			// 	for (size_t j = 0; j < inquire[i].size(); j++)
			// 	{
			// 		cout<<inquire[i][j]<<" ";
			// 	}
			// 	cout<<endl;

			// }
		}

		myfile.close(); // close the file
	}
	else
	{
		std::cout << "Unable to open file " << instance_name << std::endl
				  << "Please check the file location or file name." << std::endl; // throw an error message
		getchar();																  // keep the console window
		exit(1);																  // end the program
	}
}
void write_result_cpu(std::string instance_name, std::vector<double> &times,std::vector<int> &costs,std::string tail)
{
	std::cout << "writes " << endl;
	std::string line_content;
	string temp = "/home/sunyahui/lijiayu/GST/build/bin/"+instance_name+".cpures"+tail;
	 std::ofstream outfile(temp); 
	 if (!outfile.is_open()) { // 检查文件是否成功打开
        std::cerr << "无法打开文件" << std::endl;
        
    }
	for (int i = 0; i < costs.size(); i++)
	{
		outfile<<times[i]<<" "<<costs[i]<<endl;
	}
	 outfile.close(); // 关闭文件
}

void write_result_gpu(std::string instance_name, std::vector<double> &times,std::vector<int> &costs,std::string tail)
{
	std::cout << "writes " << endl;
	std::string line_content;
	string temp = "/home/sunyahui/lijiayu/GST/build/bin/"+instance_name+".gpures"+tail;
	 std::ofstream outfile(temp); 
	 if (!outfile.is_open()) { // 检查文件是否成功打开
        std::cerr << "无法打开文件" << std::endl;
        
    }
	for (int i = 0; i < costs.size(); i++)
	{
		outfile<<times[i]<<" "<<costs[i]<<endl;
	}
	 outfile.close(); // 关闭文件
}

