#pragma once
#include <vector>
using namespace std;

#include "MainHeader.h"
#include "matrix.h"

RETURN_STATUS MY_File_Scan(	FILE* pFile, char* pFilePath, char* pMode,	double* pDataBuf, long* pDataSize);

void MY_File_Scan_Matrix(FILE* pFile, char* pFilePath, char* pMode,double* pDataBuf, int nRow,int i_column);