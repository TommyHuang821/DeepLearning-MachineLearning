#include <cmath>
#include "mymath.h"

void swap(int A[], int x, int y){
	int temp = A[x];
	A[x]=A[y];
	A[y]=temp;
}

/////////// min & max//////////////////////
int findminIndex(double A[], int start, int size)
{
	int minIndex=-1;
	for (int i=start; i<size;i++)
		if (minIndex<0 || A[i]<A[minIndex])
			minIndex=i;
	return minIndex;
}
double findMinValue(double A[ ],int start,int size) {
	double minValue = A[0];
	for(int i= start; i< size; i++)
		if(A[i] < minValue)
			minValue= A[i];/* save the largest value so far */
	return	minValue;
}
int findmaxIndex(double A[ ],int start, int size){
	int maxIndex=-1;
	for (int i=start; i<size;i++)
		if (maxIndex<0 || A[i]>A[maxIndex])
			maxIndex=i;
	return maxIndex;
}
double findMaxValue(double A[ ],int start,int size) {
	double maxValue= A[0];
	for(int i = start; i< size; i++)
		if(A[i] > maxValue)
			maxValue= A[i];/* save the largest value so far */
	return	maxValue;
}


////////// Statistic///////////////
double Math_Sum(double value_arrary[], long size)
{
    double sum = 0;

    for(int i = 0; i < size; i++)
	{
		sum = sum+ value_arrary[i];
	}
	return sum;
}
double Math_Mean(double value_arrary[], long size)
{
	double sum= Math_Sum(value_arrary, size);
    return (sum /size);
}
double Statistic_Variane(double value_arrary[], long size)
{
	double mean= Math_Mean(value_arrary, size);
 
    double temp = 0;
    for(int i = 0; i < size; i++)
    {
        temp += pow((value_arrary[i] - mean), 2);
    }
    return temp / (size);
}
double Statistic_SampleVariane(double value_arrary[], long size)
{
	double mean= Math_Mean(value_arrary, size);
 
    double temp = 0;
    for(int i = 0; i < size; i++)
    {
        //temp += (value_arrary[i] - mean) * (value_arrary[i] - mean) ;
		temp += pow((value_arrary[i] - mean), 2) ;
    }
    return temp / (size-1) ;
}
double Statistic_StandardDeviation(double value_arrary[], long size)
{
    return sqrt(Statistic_Variane(value_arrary, size));
}
double Statistic_SampleStandardDeviation(double value_arrary[], long size)
{
    return sqrt(Statistic_SampleVariane(value_arrary, size));
}
double Statistic_Skewness(double value_arrary[], long size)
{
	double mean= Math_Mean(value_arrary, size);
	double std= Statistic_SampleStandardDeviation(value_arrary, size);
 
    double k3 = 0;
	double k2 = 0;
    for(int i = 0; i < size; i++)
    {
        k3 += pow((value_arrary[i] - mean), 3);
		k2 += pow((value_arrary[i] - mean), 2);
    }
	k3=k3/size;
	k2=k2/size;
    //return (temp / (size))*(pow(size*(size-1),0.5)/(size-1));
	return k3/pow(k2,1.5);
}
double Statistic_Kurtosis(double value_arrary[], long size)
{
	double mean= Math_Mean(value_arrary, size);
	double std= Statistic_StandardDeviation(value_arrary, size);

    double temp = 0;
    for(int i = 0; i < size; i++)
    {
        temp += pow((value_arrary[i] - mean), 4)/pow(std,4);
    }
    return temp / (size);
}
double Statistic_Covariane(double *value_arrary1,double *value_arrary2, long size)
{
	double mean1= Math_Mean(value_arrary1, size);
	double mean2= Math_Mean(value_arrary2, size);
	double tmp=0;

	for(int i=0; i<size; i++)
	{
		tmp+=(value_arrary1[i]-mean1)*(value_arrary2[i]-mean2);	
	}
	return tmp/size;
}



// norm of a vector
double Math_pNorms(double value_arrary[], long size, int p)
{
	
	double temp = 0, pNorm=0, doublep=0;
	doublep=(1/double(p));
	if (p==1)
	{
		for(int i = 0; i < size; i++)
		{
			temp += abs(value_arrary[i]);
		}
		pNorm=temp;
	}
	else
	{
		for(int i = 0; i < size; i++)
		{
			temp += pow(abs(value_arrary[i]) , p);
		}
		pNorm=pow(temp, doublep);
	}
	return pNorm;
}
double getArraryValue(double value_arrary[], long Poisiton)
{
	return value_arrary[Poisiton];
}
// zero mean of a vecotr
void Math_Vector_ZeroMean(double *value_arrary, long size)
{
	double mean= Math_Mean(value_arrary, size);
    for(int i = 0; i < size; i++)
    {
		*(value_arrary+i)=*(value_arrary+i)-mean;
    }
}
// Euclidean distance between two vectors
double Math_Vector_EuclideanDistance(double *value_arrary1,double *value_arrary2, long dimension)
{
	double tmp=0;
    for(int i = 0; i < dimension; i++)
    {
		tmp+=pow(*(value_arrary1+i)-*(value_arrary2+i),2);
    }
	return sqrt(tmp);
}


void Statistic_Zsore(double *value_arrary,double *Mu, double *Sigma,long size)
{
	*Mu=Math_Mean(value_arrary,size);
	*Sigma=Statistic_SampleStandardDeviation(value_arrary, size);

	for (int j=0; j<size; j++)
	{
		*(value_arrary+j)=(*(value_arrary+j)-*Mu) / *Sigma;
	}

}