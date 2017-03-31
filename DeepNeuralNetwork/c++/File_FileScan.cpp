#include "File_FileScan.h"
#include "matrix.h"

RETURN_STATUS MY_File_Scan(
	FILE* pFile, 
	char* pFilePath, 
	char* pMode,
	double* pDataBuf, 
	long* pDataSize)
{
	long i = 0;
	float tmpData = 0;


	pFile = fopen(pFilePath, pMode);
	if (pFile == NULL)
		return Message_Error;

	while(fscanf(pFile, "%f ", &tmpData)!= EOF) //
    {	
		// RRI used
		 //
		*(pDataBuf+*pDataSize)= tmpData;
		(*pDataSize)++;
    }

	fclose(pFile);

	return Message_Success;

}


void  MY_File_Scan_Matrix(
	FILE* pFile, 
	char* pFilePath, 
	char* pMode,
	double* pDataBuf,
	int nRow,
	int i_column)
{
	long i = 0;
	float tmpData1 = 0, tmpData2 = 0;

	pFile = fopen(pFilePath, pMode);

	i=0;
	while(fscanf(pFile, "%f %f  ", &tmpData1 , &tmpData2)!= EOF) //
    {	
		if (i_column==1)
		*(pDataBuf+i) = tmpData1;
		else if (i_column==2)
		*(pDataBuf+i) =tmpData2;	
		
		i++;
    }

	fclose(pFile);

}


