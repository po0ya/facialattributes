/*
 * sysutils.cpp
 *
 *  Created on: Jul 16, 2015
 *      Author: pouya
 */

#include"sysutils.hpp"

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "sys/types.h"
#include "sys/sysinfo.h"
#include<iostream>
#include"confs.h"

static int parseLine(char* line) {
	int i = strlen(line);
	while (*line < '0' || *line > '9')
		line++;
	line[i - 3] = '\0';
	i = atoi(line);
	return i;
}

int getPhysMemVal() {
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmRSS:", 6) == 0) {
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return result;
}
int getVirtMemVal() {
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmSize:", 7) == 0) {
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return result;
}
int getTotalRam() {
	struct sysinfo memInfo;
	sysinfo(&memInfo);
	long long totalVirtualMem = memInfo.totalram;
	//Add other values in next statement to avoid int overflow on right hand side...
	totalVirtualMem += memInfo.totalswap;
	totalVirtualMem *= memInfo.mem_unit;
	return totalVirtualMem;

}
int getTotalVirtMem() {
	struct sysinfo memInfo;
	sysinfo(&memInfo);
	long long totalVirtualMem = memInfo.totalram;
	//Add other values in next statement to avoid int overflow on right hand side...
	totalVirtualMem += memInfo.totalswap;
	totalVirtualMem *= memInfo.mem_unit;
	return totalVirtualMem;
}

void printMemUsage() {
	std::cout << "RAM: " << getPhysMemVal() / 1000 << "MB -- Virt: "
			<< getVirtMemVal() / 1000 << "MB" << std::endl;
}

void tictacMem() {
	std::cout<<std::endl;
	printMemUsage();
	static int mem = 0;
	static bool flag = true;
	if (flag) {
		mem = getPhysMemVal();
		flag = false;
	} else {
		std::cout << "RAM used by this portion: "
				<< (getPhysMemVal() - mem) / 1000 << "MB" << std::endl;
		flag = true;
	}
	std::cout<<std::endl;

}
