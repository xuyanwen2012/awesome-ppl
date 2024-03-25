#include <fstream>
#include <iostream>
#include <unordered_map>

#include "yaml-cpp/yaml.h"

struct StageConfig
{
	std::string deviceType; // CPU or GPU
	int numThreads; // Number of threads if CPU
	int gridSize; // Grid size if GPU
	int blockSize; // Block size if GPU
};

int main()
{
	std::unordered_map<int, StageConfig>
		configMap; // Hashtable to store configurations

	// Load YAML file
	YAML::Node config = YAML::LoadFile("config.yaml");

	// Iterate through stages
	for (int i = 1; i <= 6; ++i)
	{
		// Read stage configuration
		YAML::Node stageNode = config["Stage" + std::to_string(i)];
		if (!stageNode)
		{
			std::cerr << "Error: Stage " << i << " not found in configuration file."
				<< std::endl;
			continue;
		}

		StageConfig stageConfig;
		stageConfig.deviceType = stageNode["DeviceType"].as<std::string>();

		// Parse device-specific parameters
		if (stageConfig.deviceType == "CPU")
		{
			stageConfig.numThreads = stageNode["NumThreads"].as<int>();
		}
		else if (stageConfig.deviceType == "GPU")
		{
			stageConfig.gridSize = stageNode["GridSize"].as<int>();
			stageConfig.blockSize = stageNode["BlockSize"].as<int>();
		}
		else
		{
			std::cerr << "Error: Invalid device type specified for Stage " << i
				<< std::endl;
			continue;
		}

		// Store configuration in hashtable
		configMap[i] = stageConfig;
	}

	// Example: Accessing configuration for Stage 3
	if (configMap.find(3) != configMap.end())
	{
		StageConfig& stage3Config = configMap[3];
		std::cout << "Stage 3 Device Type: " << stage3Config.deviceType
			<< std::endl;
		if (stage3Config.deviceType == "CPU")
		{
			std::cout << "Number of Threads: " << stage3Config.numThreads
				<< std::endl;
		}
		else
		{
			std::cout << "Grid Size: " << stage3Config.gridSize << std::endl;
			std::cout << "Block Size: " << stage3Config.blockSize << std::endl;
		}
	}
	else
	{
		std::cerr << "Error: Configuration for Stage 3 not found." << std::endl;
	}

	return 0;
}
