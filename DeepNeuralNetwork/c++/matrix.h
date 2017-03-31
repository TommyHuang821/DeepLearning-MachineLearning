// Matrix Relation
// header file for matrix template class
// NOTE:  
// 
// write by Chih-Sheng (Tommy) Huang, 12/21/2016 
// version 1.0

/* 
/////  Basic function list /////
	void allocate();
	matrix(int newmaxsize); //Square matrix 
	matrix(int newmaxsize, int newactualsize);  // Matrix(newmaxsize x newactualsize)
	~matrix(); // delete
	void comparetoidentity();
	void copymatrix(matrix&  source);
	void setactualsize(int newactualsize) ;
	int getactualsize() { return actualsize; };
	int getmaxsize() { return maxsize; };
	void getvalue(int row, int column, D& returnvalue, bool& success) ;
	bool setvalue(int row, int column, D newvalue) ;

/////////////// Matrix caluation///////////////////
	void invert(); // Matrix inverse
	void transposematrix(matrix&  source); // Matrix tranpose 
	void MultiplyConstant(double a);

	void settoproduct(matrix& left, matrix& right); //Matrix product, right * left
	void settoplus(matrix& A, matrix& B); // Matrix plus B+A;
	void settominus(matrix& A, matrix& B); // Matrix minus A-B;

	void CovarianceMatrix(matrix& X);  // Covariance matrix, X (size: n * dim);
	void eigenvalue_dim2(matrix& A)// A must be squred (2*2);//
	void eigenvector_dim2(matrix& A, matrix& Eigvalue);//

//////////////////////////////////////////////////

Example: 
	matrix <double> M1(200,200);  // for test we create & invert this matrix
	matrix <double> M2(4,3);      // this will be a copy of original M1
	matrix <double> M3(3,2);      // this will contain the product
	matrix <double> X(100,2);
	matrix <double> XT(2,100);

	M2.copymatrix(M1);
	M1.invert();  // invert the matrix
	M3.settoproduct(M1,M2);//product: original * inverse 
	M3.settoproduct(M2,M1); //product: inverse * original
	
	XT.transposematrix(X); // transpose
	CovM.CovarianceMatrix(X); // 2 * 2 covariance matrix
*/



#ifndef __mjdmatrix_h
#define __mjdmatrix_h
#include <iostream>

// generic object (class) definition of matrix:
template <class D> class matrix
{
  // NOTE: maxsize determines available memory storage, but
  // actualsize determines the actual size of the stored matrix in use
  // at a particular time.
public:
	// NOTE: maxsize determines available memory storage, but
  // actualsize determines the actual size of the stored matrix in use
  // at a particular time.
  int maxsize;  // max number of rows (same as max number of columns)
  int actualsize;  // actual size (rows, or columns) of the stored matrix
  D* data;      // where the data contents of the matrix are stored

  void allocate()   
  {
    delete[] data;
    data = new D [maxsize*actualsize];
  };

  matrix() {matrix(2,2);};                  // private ctor's
  matrix(int newmaxsize) {matrix(newmaxsize,newmaxsize);};


	matrix(int newmaxsize, int newactualsize)  
	{ // the only public ctor
		/*
		if (newmaxsize <= 0) newmaxsize = 5;
			maxsize = newmaxsize; 
		if ((newactualsize <= newmaxsize)&&(newactualsize>0))
			actualsize = newactualsize;
		else 
			actualsize = newmaxsize;
		// since allocate() will first call delete[] on data:
		*/

		maxsize=newmaxsize;
		actualsize=newactualsize;
		data = 0;
		allocate();
    };

	~matrix() { delete[] data; };
	
	void setresize(int newmaxsize,int newactualsize)
	{
		maxsize=newmaxsize;
		actualsize=newactualsize;
		data = 0;
		allocate();
	}
	
	
	
	void comparetoidentity()  
	{
		int worstdiagonal = 0;
		D maxunitydeviation = 0.0;
		D currentunitydeviation;
		for ( int i = 0; i < actualsize; i++ )  
		{
			currentunitydeviation = data[i*maxsize+i] - 1.;
			if ( currentunitydeviation < 0.0) currentunitydeviation *= -1.;
			if ( currentunitydeviation > maxunitydeviation )  
			{
				maxunitydeviation = currentunitydeviation;
				worstdiagonal = i;
			}
		}

		int worstoffdiagonalrow = 0;
		int worstoffdiagonalcolumn = 0;
		D maxzerodeviation = 0.0;
		D currentzerodeviation ;

		for ( int i = 0; i < maxsize; i++ )  
		{
			for ( int j = 0; j < actualsize; j++ )  
			{
				if ( i == j ) continue;  // we look only at non-diagonal terms
				currentzerodeviation = data[i*maxsize+j];
				if ( currentzerodeviation < 0.0) currentzerodeviation *= -1.0;
				if ( currentzerodeviation > maxzerodeviation )  
				{
					maxzerodeviation = currentzerodeviation;
					worstoffdiagonalrow = i;
					worstoffdiagonalcolumn = j;
				}

			}
		}
		cout << "Worst diagonal value deviation from unity: " 
			<< maxunitydeviation << " at row/column " << worstdiagonal << endl;
		cout << "Worst off-diagonal value deviation from zero: " 
			<< maxzerodeviation << " at row = " << worstoffdiagonalrow 
			<< ", column = " << worstoffdiagonalcolumn << endl;
	}

	void settoproduct(matrix& left, matrix& right)  
	{
		actualsize = left.getactualsize();
		long nElement=left.getmaxsize();
		for ( int i = 0; i < maxsize; i++ )
			for ( int j = 0; j < actualsize; j++ )  
			{
				D sum = 0.0;
				D leftvalue, rightvalue;
				bool success;
				for (int c = 0; c < nElement; c++)  
				{
					right.getvalue(i,c,rightvalue,success);
					left.getvalue(c,j,leftvalue,success);
					sum += leftvalue * rightvalue;
				}
			setvalue(i,j,sum);
			}
    }


	void MultiplyConstant(double a)
	{
		for ( int i = 0; i < maxsize; i++ )
			for ( int j = 0; j < actualsize; j++ )  
			{
				data[i*maxsize+j] = a*data[i*maxsize+j];
			}
	
	}


	void copymatrix(matrix&  source)  
	{
		actualsize = source.getactualsize();
		maxsize = source.getmaxsize();

		for ( int i = 0; i < maxsize; i++ )
			for ( int j = 0; j < actualsize; j++ )  
			{
				D value;
				bool success;
				source.getvalue(i,j,value,success);
				data[i*maxsize+j] = value;
			}
	};


	void transposematrix(matrix&  source)  
	{
		maxsize=source.getactualsize();// because transpose
		actualsize=source.getmaxsize();// because transpose


		for ( int i = 0; i < maxsize; i++ )
			for ( int j = 0; j < actualsize; j++ )  
			{
				D value;
				bool success;
				source.getvalue(j,i,value,success);
				//source.setvalue(j,i,value);
				data[j+i*actualsize] = value;

			}
	};


	void setactualsize(int newactualsize) 
	{
		if ( newactualsize > maxsize )
			{
				maxsize = newactualsize ; // * 2;  // wastes memory but saves
												// time otherwise required for
												// operation new[]
				allocate();
			}
		if (newactualsize >= 0) actualsize = newactualsize;
	};

	int getactualsize() { return actualsize; };
	int getmaxsize() { return maxsize; };

	void getvalue(int row, int column, D& returnvalue, bool& success)   
	{
		if ( (row<0) || (column<0) )
			{  success = false;
				return;    }
		returnvalue = data[ row * actualsize + column ];
		
		success = true;
	};

	bool setvalue(int row, int column, D newvalue)  
	{
		if ( (row<0) || (column<0) ) return false;

		data[ row * actualsize + column ] = newvalue;
		

		return true;
	};

	void invert()  
	{
		if (actualsize <= 0) return;  // sanity check
		if (actualsize == 1) return;  // must be of dimension >= 2
		for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
		for (int i=1; i < actualsize; i++)  { 
			for (int j=i; j < actualsize; j++)  { // do a column of L
			D sum = 0.0;
			for (int k = 0; k < i; k++)  
				sum += data[j*maxsize+k] * data[k*maxsize+i];
			data[j*maxsize+i] -= sum;
			}
			if (i == actualsize-1) continue;
			for (int j=i+1; j < actualsize; j++)  {  // do a row of U
			D sum = 0.0;
			for (int k = 0; k < i; k++)
				sum += data[i*maxsize+k]*data[k*maxsize+j];
			data[i*maxsize+j] = 
				(data[i*maxsize+j]-sum) / data[i*maxsize+i];
			}
			}
		for ( int i = 0; i < actualsize; i++ )  // invert L
			for ( int j = i; j < actualsize; j++ )  {
			D x = 1.0;
			if ( i != j ) {
				x = 0.0;
				for ( int k = i; k < j; k++ ) 
					x -= data[j*maxsize+k]*data[k*maxsize+i];
				}
			data[j*maxsize+i] = x / data[j*maxsize+j];
			}
		for ( int i = 0; i < actualsize; i++ )   // invert U
			for ( int j = i; j < actualsize; j++ )  {
			if ( i == j ) continue;
			D sum = 0.0;
			for ( int k = i; k < j; k++ )
				sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
			data[i*maxsize+j] = -sum;
			}
		for ( int i = 0; i < actualsize; i++ )   // final inversion
			for ( int j = 0; j < actualsize; j++ )  {
			D sum = 0.0;
			for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
				sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
			data[j*maxsize+i] = sum;
			}
	};


	void CovarianceMatrix(matrix& X)
	{
		long nElement=X.getmaxsize();
		long dim=X.getactualsize();	
		
		matrix <double> XT(dim,nElement); 
		matrix <double> CovM(dim,dim); 
		XT.transposematrix(X); 
		CovM.settoproduct(X,XT);

		for ( int i = 0; i < dim; i++ )
			for ( int j = 0; j < dim; j++ )  
			{
				D value;
				bool success;
				CovM.getvalue(i,j,value,success);
				data[i*dim+j] = value/nElement;
			}
	}

	void settoplus(matrix& A, matrix& B)  
	{
		if ((A.getactualsize()!=B.getactualsize()) &&
		(A.getmaxsize()!=B.getmaxsize()))
		{  
			allocate();
			return;
		}

		for ( int i = 0; i < A.getmaxsize(); i++ )
			for ( int j = 0; j < A.getactualsize(); j++ )  
			{
				bool success;
				D leftvalue, rightvalue;
				A.getvalue(i,j,rightvalue,success);
				B.getvalue(i,j,leftvalue,success);
				setvalue(i,j,leftvalue + rightvalue);
			}	
    }

	void settominus(matrix& A, matrix& B)  
	{
		if ((A.getactualsize()!=B.getactualsize()) &&
		(A.getmaxsize()!=B.getmaxsize()))
		{  
			allocate();
			return;
		}

		for ( int i = 0; i < A.getmaxsize(); i++ )
			for ( int j = 0; j < A.getactualsize(); j++ )  
			{
				bool success;
				D leftvalue, rightvalue;
				A.getvalue(i,j,rightvalue,success);
				B.getvalue(i,j,leftvalue,success);
				setvalue(i,j,rightvalue - leftvalue);
			}	
    }

	void eigenvalue_dim2(matrix& A)
	{
		if ((A.getactualsize()!=A.getmaxsize()) && (A.getactualsize()!=2) && (A.getmaxsize()!=2)) 
		{return;}
		
		double a=0,b=0,c=0,tmpE1=0,tmpE2=0;
		a=1;
		b=-(A.data[0]+A.data[3]);
		c=A.data[0]*A.data[3]-A.data[1]*A.data[2];
		tmpE1=(-b+sqrt(pow(b,2)-4*a*c))/(2*a);
		tmpE2=(-b-sqrt(pow(b,2)-4*a*c))/(2*a);
		
		setvalue(0,1,0);	
		setvalue(1,0,0);
		if (tmpE1>=tmpE2)
		{
			setvalue(0,0,tmpE1);	
			setvalue(1,1,tmpE2);	
		}
		else
		{
			setvalue(0,0,tmpE2);	
			setvalue(1,1,tmpE1);
		}
	}

	void eigenvector_dim2(matrix& A, matrix& Eigvalue)
	{
		matrix <double> LandaI(2,2);
		double tmp1=0,tmp2=0,tmp=0;
		for (int i=0; i<2;i++)
		{
			D value;
			bool sucess;
			Eigvalue.getvalue(i,i,value, sucess);
			LandaI.data[0]=1*value;
			LandaI.data[1]=0;
			LandaI.data[2]=0;
			LandaI.data[3]=1*value;
		
			LandaI.settominus(A, LandaI) ;

			tmp=0;
			tmp=-(LandaI.data[1]-LandaI.data[3])/(LandaI.data[0]-LandaI.data[2]);
			if (i==0)
			{
				data[0]=tmp;
				data[1]=1;
			}
			else
			{
				data[2]=1;
				data[3]=tmp;
			}
		}
		LandaI.data[0]=data[0];
		LandaI.data[1]=data[1];
		LandaI.data[2]=data[2];
		LandaI.data[3]=data[3];

		LandaI.eigenvalue_dim2(LandaI);
		tmp=LandaI.data[0];

		data[0]=data[0]/tmp;
		data[1]=data[1]/tmp;
		data[2]=data[2]/tmp;
		data[3]=data[3]/tmp;

	}

};
#endif
