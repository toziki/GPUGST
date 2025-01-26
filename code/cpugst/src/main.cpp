#include <iostream>
#include <fstream>

using namespace std;

// header files in the Boost library: https://www.boost.org/
#include <boost/random.hpp>
boost::random::mt19937 boost_random_time_seed{static_cast<std::uint32_t>(std::time(0))};
#include "exp_CPU_nonHop.h"

int main(int argc, char *argv[])
{
	cout << "Start running..." << endl;
	auto begin = std::chrono::high_resolution_clock::now();

	if (atoi(argv[1]) == 1)

	{ // argv[0] is the name of the exe file  e.g.: ./A musae musae 50 3600 // 1: exp_CPU_nonHop
		cout << argv[2] << " " << argv[3] << " " << argv[4] << " " << argv[5] << " " << argv[6] << " " << endl;
		exp_CPU_nonHop(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
	}

	auto end = std::chrono::high_resolution_clock::now();
	double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	cout << "END    runningtime: " << runningtime << "s" << endl;
}