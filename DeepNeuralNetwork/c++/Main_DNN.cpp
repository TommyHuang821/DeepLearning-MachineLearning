// SWM_DNN.cpp : 定義主控台應用程式的進入點。
//


#include <iostream>
#include <time.h>
#include "DNN_Learning.h"
#include "MainHeader.h"
#include "File_FileScan.h"
#include "matrix.h"
#include "mymath.h"

using namespace std;

// 
int dim=2; // dimension of the input data
double DataBuf_TrainData[2][400]; // allocate the training data
double DataBuf_TrainOut[400]={0}; // allocate the training label
double DataBuf_X[3][400]; // add the bias to train data
double DataBuf_TestData[2][400]={0}; // allocate the testing data
double DataBuf_TestOut[400]={0}; // allocate the testing label
long kk[400]={0}; // the random arrary for the batch index of trining data
int dataSize_Train = 400; // size of the training data
int dataSize_Test = 400; // size of the testing data
char* pFilePath_Train="Trainingdata.txt";
char* pFilePath_TrainOut="TrainingOut.txt";
long pDataSize=0; //
FILE* pFile = NULL;
// for Z-score
double Mu_train[2]={0};
double Std_train[2]={0};
double Mu_out=0;
double Std_out=0;



int main()
{

	for (int i=0; i<2; i++)
	MY_File_Scan_Matrix(pFile,pFilePath_Train, "rb",DataBuf_TrainData[i],dataSize_Train,(i+1));

	pFile = NULL;
	MY_File_Scan(pFile, pFilePath_TrainOut, "rb",	DataBuf_TrainOut, &pDataSize);



	long tmp_count=0;
	double errorvalue=0;
	
	// parameter for NN structure
	int SizeInputLayer=dim+1; // number of nodes of input layer 
	int SizeHiddenLayer_1=10;  // number of nodes first hidden layer
	int SizeHiddenLayer_2=10;  // number of nodes second hidden layer
	int SizeOutputLayer=1;    // number of nodes output layer
	double LearningRate=0.01; // learning rate 
	int maxIter=2000;
	int batchsize=100;
	char* AF1="T"; // AactivationFunction for input to hidden layer 1
	char* AF2="S"; // AactivationFunction for hidden layer 1 to hidden layer 2
	char* AF3="L"; // AactivationFunction for hidden layer 2 to output layer (must be linear)
		// T: Tanh
		// S: Sigmoid
		// R: ReLU,
		// L: Linear
	double *CostValue; // NN cost value
	CostValue = new double [maxIter];

	// initial NN weight
	DNN_Net_Strutcure DNN_net_in;
	Set_NN_initial_Weight( &DNN_net_in, SizeHiddenLayer_1, SizeInputLayer,LearningRate, batchsize, *AF1 );
	DNN_Net_Strutcure DNN_net_1;
	Set_NN_initial_Weight( &DNN_net_1, SizeHiddenLayer_2, SizeHiddenLayer_1,LearningRate,batchsize, *AF2 );
	DNN_Net_Strutcure DNN_net_out;
	Set_NN_initial_Weight( &DNN_net_out, SizeOutputLayer, SizeHiddenLayer_2,LearningRate, batchsize, *AF3);
	

	// Zscore
	for (int i=0; i<dim; i++) 
	{
		Statistic_Zsore(DataBuf_TrainData[i], &Mu_train[i], &Std_train[i], dataSize_Train);
	}
	Statistic_Zsore(DataBuf_TrainOut, &Mu_out, &Std_out, dataSize_Train);


	// add the bias
	for (int j=0;j<dataSize_Train;j++)
	{
		for (int i=0; i<dim; i++) 
		{
			DataBuf_X[i][j]=DataBuf_TrainData[i][j];
			
		}
		DataBuf_X[dim][j]=1;
	}
	
	double numbatches= dataSize_Train / batchsize;
	

	matrix <double> *batch_x;
	matrix <double> *batch_x_T;
	matrix <double> *batch_x_out;
	batch_x= (matrix <double> *) malloc( sizeof(matrix <double> *));
	batch_x_T= (matrix <double> *) malloc( sizeof(matrix <double> *));
	batch_x_out= (matrix <double> *) malloc( sizeof(matrix <double> *));

    batch_x->maxsize=(dim+1);
	batch_x->actualsize=batchsize;
	batch_x->setresize(batch_x->maxsize,batch_x->actualsize);
	batch_x_T->maxsize=batchsize;
	batch_x_T->actualsize=(dim+1);
	batch_x_T->setresize(batch_x_T->maxsize,batch_x_T->actualsize);
	batch_x_out->maxsize=1;
	batch_x_out->actualsize=batchsize;
	batch_x_out->setresize(batch_x_out->maxsize,batch_x_out->actualsize);
	
	
	
	//

	//clock_t start, end;
    //double cpu_time_used;
     
    

	for (int iter=0; iter<maxIter;iter++)
	{		
		Random_Permutation(kk, dataSize_Train);	
		errorvalue=0;
		//start = clock();
		for (int ibatch=0;ibatch<int(numbatches);ibatch++)
		{
			// sub-batch data
			tmp_count=0;
			for (int j= (ibatch * batchsize ); j< ((ibatch+1) * batchsize);j++)	
			{
				for (int i=0; i<dim+1;i++)
				{
					batch_x->setvalue(i,tmp_count,DataBuf_X[i][kk[j]]); // batch train data
				}
				batch_x_out->setvalue(0,tmp_count,DataBuf_TrainOut[kk[j]]);//batch train label
				tmp_count++;
			}
			//
			DNN_net_in.weight_T->transposematrix(*(DNN_net_in.weight));
			DNN_net_1.weight_T->transposematrix(*(DNN_net_1.weight));
			DNN_net_out.weight_T->transposematrix(*(DNN_net_out.weight));



			
			/////////feedforward ////////
			//Layer 1
			DNN_fordward(&DNN_net_in,batch_x);
			//Layer 2
			DNN_fordward(&DNN_net_1,DNN_net_in.V);
			//Layer 3
			DNN_fordward(&DNN_net_out,DNN_net_1.V);
			///////////////////////////////

			/////////////back-propagation/////
			DNN_net_out.Err ->settominus(*DNN_net_out.V,*batch_x_out);
			// Layer3->layer2
			DNN_backpropagation(&DNN_net_out,&DNN_net_1);
			// Layer2->layer1
			DNN_backpropagation(&DNN_net_1,&DNN_net_in);
			
			//不知道為什麼這行執行完，DNN_net3.weight位置改了
			//DNN_net_out.weight->transposematrix(*DNN_net_out.weight_T);
			//
			batch_x_T->transposematrix(*batch_x);
			DNN_backpropagation_inputlayer(&DNN_net_in,batch_x_T);
			//
			DNN_net_in.weight->settominus(*DNN_net_in.weight,*DNN_net_in.delta_weight);
			DNN_net_1.weight->settominus(*DNN_net_1.weight,*DNN_net_1.delta_weight);
			DNN_net_out.weight->settominus(*DNN_net_out.weight,*DNN_net_out.delta_weight);
			//////////////////////////////////
			

			errorvalue+=DNN_predictError(&DNN_net_out, batch_x_out);

		}
		//end = clock();
		//cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

		CostValue[iter]=errorvalue;

	
		printf("Inter: %d ,Cost: %f\n", iter, CostValue[iter]);
	}
	
	return 0;
}