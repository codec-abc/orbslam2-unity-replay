#include <cassert>

#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <limits>
#include <sstream>
#include <iostream>
using namespace std;

int readStringFileAndOutputBin()
{
	auto filename = "C:\\DEV\\orb_slam2_windows\\Vocabulary\\ORBvoc.txt";
	ifstream f;
	f.open(filename);

	int line_number = 1;

	if (f.eof())
		return -2;

	vector<int> lines;

	string s;
	getline(f, s);
	stringstream ss;

	ss << s;

	int m_k;
	int m_L;
	int n1, n2;

	ss >> m_k;
	ss >> m_L;
	ss >> n1;
	ss >> n2;

	lines.push_back(m_k);
	lines.push_back(m_L);
	lines.push_back(n1);
	lines.push_back(n2);

	if (m_k < 0 || m_k>20 || m_L < 1 || m_L>10 || n1 < 0 || n1>5 || n2 < 0 || n2>3)
	{
		cout << "Vocabulary loading failure: This is not a correct text file!" << endl;
		return -1;
	}

	while (!f.eof())
	{
		string snode;
		getline(f, snode);
		line_number++;
		stringstream ssnode;
		ssnode << snode;

		int pid;
		ssnode >> pid;
		lines.push_back(pid);

		int nIsLeaf;
		ssnode >> nIsLeaf;
		lines.push_back(nIsLeaf);

		stringstream ssd;
		for (int iD = 0; iD < 32; iD++)
		{
			int sElement;
			ssnode >> sElement;
			lines.push_back(sElement);
		}

		float weight;
		ssnode >> weight;
		int* new_int_ptr = reinterpret_cast<int*>(&weight);
		int new_int = *new_int_ptr;
		lines.push_back(new_int);

		//cout << "weight as int " << new_int << endl;
		//float* new_float_ptr = reinterpret_cast<float*>(&new_int);
		//float new_float = *new_float_ptr;
		//cout << "weight as float casted back " << new_float << endl;

	}

	std::ofstream outfile("vocabulary.bin", ios::out | std::ofstream::binary);
	//outfile.write(buffer, size);

	outfile.write(reinterpret_cast<const char*>(&lines[0]), lines.size() * sizeof(int));

	cout << "end" << endl;
	return 0;
}


int readFile()
{
	std::ifstream is("vocabulary.bin", std::ifstream::binary);
	if (!is)
	{
		return -1;
	}
	// Determine the file length
	is.seekg(0, ios_base::end);
	std::size_t size = is.tellg();
	is.seekg(0, std::ios_base::beg);
	
	// Create a vector to store the data
	std::vector<int> v (size / sizeof(int));
	// Load the data

	is.read((char*)&v[0], size);
	// Close the file
	is.close();

	cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << endl;

	int vector_size = v.size();
	for (int i = 4; i < v.size();)
	{
		int pid = v[i];
		i++;
		int isLeaf = v[i];
		i++;
		vector<int> factors;
		for (int iD = 0; iD < 32; iD++)
		{
			factors.push_back(v[i]);
			i++;
		}
		int weight = v[i];
		i++;

		/*
		float* new_float_ptr = reinterpret_cast<float*>(&weight);
		float new_float = *new_float_ptr;
		*/
	}
	return 0;
}

int main()
{
	//return readStringFileAndOutputBin();

	return readFile();
}

