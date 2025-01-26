
#include <parse_string.h>
#include <fstream>
#include <iostream>
#include <iterator>
 #include<sstream>
 typedef std::vector<std::vector<std::pair<int,int>>> graph_v_of_v_idealID;
 using namespace std;
void read_Group(std::string instance_name, graph_v_of_v_idealID &group_graph)
{

	std::string line_content;
	string temp = instance_name;
	cout<<"group open "<<temp<<endl;
	group_graph.clear();
	std::ifstream myfile(temp); // open the file
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
			std::stringstream ss(Parsed_content[1]);
			
			std::istream_iterator<std::string> begin(ss);
			std::istream_iterator<std::string> end;
			std::vector<std::string> groups_mem(begin, end);
			group_graph[g].resize(groups_mem.size());
			for (size_t i = 0; i < groups_mem.size(); i++)
			{
				
				int v = std::stoi(groups_mem[i]);
		
				group_graph[g][i] = {v, 1};
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
void read_inquire(std::string instance_name, std::vector<std::vector<int>> &inquire)
{
	std::cout << "open inquire" << " ";
	inquire.clear();
	std::string line_content;
	string temp = instance_name;
	cout<<"inquire open "<<temp<<endl;
	std::ifstream myfile(temp); // open the file
	if (myfile.is_open())				 // if the file is opened successfully
	{
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
void write_result(std::string instance_name, std::vector<double> &times,std::vector<int> &costs)
{
	std::cout << "writes " << endl;
	std::string line_content;
	string temp = "/home/sunyahui/lijiayu/simd/data/"+instance_name+"/"+instance_name+".res7";
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
