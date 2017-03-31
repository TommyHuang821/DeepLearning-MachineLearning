
#pragma once
#include "DNN_Learning.h"


// Random Permutation
int myrandom (int i) { return std::rand()%i;}
void Random_Permutation(long* d64outputBuffer, long i32num)
{
	int k = 0 ;
		std::srand ( unsigned ( std::time(0) ) );
		std::vector<int> myvector;

		// set some values:
		for (int i = 1; i < i32num+1; i++) myvector.push_back(i);

		// using built-in random generator:
		std::random_shuffle ( myvector.begin(), myvector.end() );

		// using myrandom:
		std::random_shuffle ( myvector.begin(), myvector.end(), myrandom);

		// print out content:
		for (std::vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
		{
		d64outputBuffer[k] = *it;
		k++;
	}
}


void Set_NN_initial_Weight(DNN_Net_Strutcure *p, int L1, int L2, double LearningRate ,int batch, char AactivationFunction)
{
	p->AactivationFunction=AactivationFunction;
	p->L1 = L1;
	p->L2 = L2;
	p->batch=batch;
	p->LearningRate=LearningRate;
	p->weight= (matrix <double> *) malloc( sizeof(matrix <double> *));
	p->weight_T= (matrix <double> *) malloc( sizeof(matrix <double> *));
	p->delta_weight= (matrix <double> *) malloc( sizeof(matrix <double> *));
	p->V= (matrix <double> *) malloc( sizeof(matrix <double> *));
	p->V_T= (matrix <double> *) malloc( sizeof(matrix <double> *));
	p->Z= (matrix <double> *) malloc( sizeof(matrix <double> *));
	p->Err= (matrix <double> *) malloc( sizeof(matrix <double> *));
	p->tmpvalue= (matrix <double> *) malloc( sizeof(matrix <double> *));
	

	p->weight->maxsize=L1;
	p->weight->actualsize=L2;
	p->weight->setresize(L1,L2);

	p->weight_T->maxsize=L2;
	p->weight_T->actualsize=L1;
	p->weight_T->setresize(L2,L1);

	p->delta_weight->maxsize=L1;
	p->delta_weight->actualsize=L2;
	p->delta_weight->setresize(L1,L2);

	p->V->maxsize=L1;
	p->V->actualsize=batch;
	p->V->setresize(L1,batch);

	p->V_T->maxsize=batch;
	p->V_T->actualsize=L1;
	p->V_T->setresize(batch,L1);

	p->Z->maxsize=L1;
	p->Z->actualsize=batch;
	p->Z->setresize(L1,batch);

	p->Err->maxsize=L1;
	p->Err->actualsize=batch;
	p->Err->setresize(L1,batch);

	p->tmpvalue->maxsize=L1;
	p->tmpvalue->actualsize=batch;
	p->tmpvalue->setresize(L1,batch);

	double tmp_w=0; 	
	for (int i=0;i<L1;i++)
	{
		for (int j=0; j<L2;j++)
		{
			tmp_w = (rand()) / (RAND_MAX + 1.0);
			tmp_w=(tmp_w-0.5)*2;
			p->weight->setvalue(i,j,tmp_w);
		}
	}
};



// feedforward
void DNN_fordward(DNN_Net_Strutcure *DNN_net,matrix <double> *batch_x)
{
	DNN_net->Z->settoproduct(*batch_x,*((*DNN_net).weight));
	
	int row,column;
	double returnvalue;
	bool sucess;
	row=DNN_net->Z->getmaxsize();
	column=DNN_net->Z->getactualsize();
	for (int i=0;i<row;i++)
		for (int j=0;j<column;j++)
		{
			DNN_net->Z->getvalue(i,j,returnvalue,sucess);
			DNN_net->inputvalue=returnvalue;
			returnvalue=DNN_net->af();
			DNN_net->V->setvalue(i,j,returnvalue);
		}	
}

// back-propagation
void DNN_backpropagation(DNN_Net_Strutcure *DNN_net2,DNN_Net_Strutcure *DNN_net1)
{		

	int row;
	int column;
	double returnvalue1,returnvalue2;
	bool sucess;
	row=DNN_net2->Err->getmaxsize();
	column=DNN_net2->Err->getactualsize();
	for (int i=0;i<row;i++)
		for (int j=0;j<column;j++)
		{
			DNN_net2->V->getvalue(i,j,returnvalue2,sucess);
			DNN_net2->inputvalue=returnvalue2;
			returnvalue2=DNN_net2->daf();
			DNN_net2->Err->getvalue(i,j,returnvalue1,sucess);
			DNN_net2->tmpvalue->setvalue(i,j,returnvalue1*returnvalue2);
		}	

	DNN_net2->tmpvalue->MultiplyConstant(DNN_net2->LearningRate); //r*delta_weight
	
	DNN_net1->V_T->transposematrix(*(DNN_net1->V));
	DNN_net2->weight_T->transposematrix(*(DNN_net2->weight));


	DNN_net1->Err->settoproduct(*((*DNN_net2).tmpvalue),*((*DNN_net2).weight_T));
	DNN_net2->delta_weight->settoproduct(*((*DNN_net1).V_T),*((*DNN_net2).tmpvalue));
}

// back-propagation
void DNN_backpropagation_inputlayer(DNN_Net_Strutcure *DNN_net,matrix <double> *batch_x_T)
{		

	int row;
	int column;
	double returnvalue1,returnvalue2;
	bool sucess;
	row=DNN_net->Err->getmaxsize();
	column=DNN_net->Err->getactualsize();
	for (int i=0;i<row;i++)
		for (int j=0;j<column;j++)
		{
			DNN_net->V->getvalue(i,j,returnvalue2,sucess);
			DNN_net->inputvalue=returnvalue2;
			returnvalue2=DNN_net->daf();
			DNN_net->Err->getvalue(i,j,returnvalue1,sucess);
			DNN_net->tmpvalue->setvalue(i,j,returnvalue1*returnvalue2);

		}	

	DNN_net->tmpvalue->MultiplyConstant(DNN_net->LearningRate); //r*delta_weight
	DNN_net->delta_weight->settoproduct(*batch_x_T,*(DNN_net->tmpvalue));

}

// predict error
double DNN_predictError(DNN_Net_Strutcure *DNN_net,matrix <double> *batch_x_out)
{		

	int row;
	int column;
	double returnvalue1,returnvalue2,tmp=0;
	bool sucess;
	row=DNN_net->V->getmaxsize();
	column=DNN_net->V->getactualsize();
	for (int i=0;i<row;i++)
		for (int j=0;j<column;j++)
		{
			DNN_net->V->getvalue(i,j,returnvalue1,sucess);
			batch_x_out->getvalue(i,j,returnvalue2,sucess);
			tmp+=pow(returnvalue1-returnvalue2,2);
		}
	return pow(tmp,0.5);	
}