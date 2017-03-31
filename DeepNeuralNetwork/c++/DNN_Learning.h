
#pragma once
#include "matrix.h"
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include "matrix.h"

typedef struct _DNN_Strutcure
{
	int L1;
	int L2;
	int batch;
	double LearningRate;
	double inputvalue;

	matrix <double> *weight;
	matrix <double> *weight_T;
	matrix <double> *delta_weight;
	matrix <double> *V; // active value
	matrix <double> *V_T; // active value
	matrix <double> *Z; //project value
	matrix <double> *Err;// error term
	matrix <double> *tmpvalue;// tmpvalue term

	char AactivationFunction;

	double af()
	{
		if (AactivationFunction =='S')
			return (1/(1+exp(-inputvalue)));
		else if (AactivationFunction =='L')
			return inputvalue;
		else if (AactivationFunction =='T')
			return ((exp(inputvalue)-exp(-inputvalue))/(exp(inputvalue)+exp(-inputvalue)));
		else if (AactivationFunction =='R')
		{	
			double tmp=0;
			if (inputvalue>=0)
				tmp=1;

			return tmp*inputvalue;
		}
	}


	double daf()
	{
		if (AactivationFunction =='S')
			return (1-inputvalue)*inputvalue;
		else if (AactivationFunction =='L')
			return 1;
		else if (AactivationFunction =='T')
			return (1-pow(inputvalue,2));
		else if (AactivationFunction =='R')
		{	
			double tmp=0;
			if (inputvalue>=0)
				tmp=1;

			return tmp;
		}
	}

}DNN_Net_Strutcure;


void Set_NN_initial_Weight(DNN_Net_Strutcure *p, int L1, int L2, double LearningRate ,int batch, char AactivationFunction);


void Random_Permutation(long* d64outputBuffer, long i32num);
void DNN_fordward(DNN_Net_Strutcure *DNN_net,matrix <double> *batch_x);
void DNN_backpropagation(DNN_Net_Strutcure *DNN_net2,DNN_Net_Strutcure *DNN_net1);
void DNN_backpropagation_inputlayer(DNN_Net_Strutcure *DNN_net,matrix <double> *batch_x_T);
double DNN_predictError(DNN_Net_Strutcure *DNN_net,matrix <double> *batch_x_out);

