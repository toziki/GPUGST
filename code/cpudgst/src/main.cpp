#include <iostream>
#include <fstream>

using namespace std;

// header files in the Boost library: https://www.boost.org/

#include "exp_CPU_Hop.h"

int main(int argc, char *argv[])
{
	cout << "Start running..." << endl;
	auto begin = std::chrono::high_resolution_clock::now();
	// /home/sunyahui/lijiayu/GST/data/ twitch 4 250 1999
	//exp_CPU_nonHop("/home/sunyahui/lijiayu/GST/data/", "twitch", 4, 0,300);
	if (atoi(argv[1]) == 2)
	{ // argv[0] is the name of the exe file  e.g.: ./A musae musae 50 3600 // 1: exp_CPU_nonHop
		exp_CPU_Hop(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));
	}


	auto end = std::chrono::high_resolution_clock::now();
	double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	cout << "END    runningtime: " << runningtime << "s" << endl;
}