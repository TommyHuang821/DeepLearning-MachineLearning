#ifndef __MYMATH__
#define __MYMATH__

#include <iostream>
#include <vector>

//////////// 


/// Basic math function
void swap(int A[], int x, int y); // switch x-th element of A and y-th element of A 
int findminIndex(double A[], int start, int size);
double findMinValue(double A[],int start,int size);
int findmaxIndex(double A[],int start, int size);
double findMaxValue(double A[],int start,int size);

// Vector
void Math_Vector_ZeroMean(double *value_arrary, long size);
double Math_Vector_EuclideanDistance(double *value_arrary1,double *value_arrary2, long dimension);
void Statistic_Zsore(double *value_arrary,double *Mu, double *Sigma,long size);

////////// Statistic///////////////
double Math_Sum(double value_arrary[], long size);// sum
double Math_Mean(double value_arrary[], long size);// mean
double Statistic_Variane(double value_arrary[], long size);// variance
double Statistic_SampleVariane(double value_arrary[], long size);// Sample variance
double Statistic_StandardDeviation(double value_arrary[], long size);// StandardDeviation
double Statistic_SampleStandardDeviation(double value_arrary[], long size);//Sample StandardDeviation
double Statistic_Kurtosis(double value_arrary[], long size); //Kurtosis
double Statistic_Skewness(double value_arrary[], long size); //Skewness
double Statistic_Covariane(double *value_arrary1,double *value_arrary2, long size);// covariance

//// LinearAlgebra//////
double Math_pNorms(double value_arrary[], long size, int p);


#endif 

