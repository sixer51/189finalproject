//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round Final: conjugate gradient solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
using namespace std;

//////////////////////////////////////////////////////////////////////////
////TODO 0: Please replace the following strings with your team name and author names
////Note: Please do not use space in the string, use "_" instead
//////////////////////////////////////////////////////////////////////////

namespace name
{
	std::string team="new 5+6";
	std::string author_1="Yijia_Wu";
	std::string author_2="Ziyue_Liu";
	std::string author_3="Xiangxin_Kong";
};

//////////////////////////////////////////////////////////////////////////
////This project implements the conjugate gradient solver to solve sparse linear systems
////For the mathematics, please read https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
////The algorithm we are implementing is in Page 50, Algorithm B.2, the standard conjugate gradient (without a preconditioner)
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
////These are the global variables that define the domain of the problem to solver (for both CPU and GPU)

const int grid_size=256;										////grid size, we will change this value to up to 256 to test your code, notice that we do not have padding elements
const int s=grid_size*grid_size;								////array size
#define I(i,j) ((i)*grid_size+(j))								////2D coordinate -> array index
#define B(i,j) (i)<0||(i)>=grid_size||(j)<0||(j)>=grid_size		////check boundary
const bool verbose=false;										////set false to turn off print for x and residual
const int max_iter_num=1000;									////max cg iteration number
const double tolerance=1e-3;									////tolerance for the iterative solver

//////////////////////////////////////////////////////////////////////////
////TODO 1: Warm up practice 1 -- implement a function for (sparse)matrix-(dense)vector multiplication and a function for vector-vector dot product

////calculate mv=M*v, here M is a square matrix
void MV(/*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,/*input vector*/const double* v,/*result*/double* mv)
{
	/*Your implementation starts*/
	for (int i = 0; i < n; i++)
		mv[i] = 0;

    for (int i = 0; i < n; i++)
    {
        for (int k = ptr[i]; k < ptr[i+1]; k++)
			mv[i] = mv[i] + val[k]*v[col[k]];

    }
    /*Your implementation ends*/
}

////return the dot product between a and b
double Dot(const double* a,const double* b,const int n)
{
	/*Your implementation starts*/
	double dp = 0.0;
	for (int i = 0; i < n; i++)
		dp += a[i] * b[i];


	/*Your implementation ends*/
	return dp;
}

//////////////////////////////////////////////////////////////////////////
////TODO 2: Warm up practice 2 -- implement a CPU-based conjugate gradient solver based on the painless PCG course notes to solve Ax=b
////Please read the notes and implement all the TODOs in the function

void Conjugate_Gradient_Solver(const double* val,const int* col,const int* ptr,const int n,		////A is an n x n sparse matrix stored in CRS format
								double* r,double* q,double* d,									////intermediate variables
								double* x,const double* b,										////x and b
								const int max_iter,const double tol)							////solver parameters
{
	////declare variables
	int iter=0;
	double delta_old=0.0;
	double delta_new=0.0;
	double alpha=0.0;
	double beta=0.0;

	double* Ax=new double[n];

	////TODO: r=b-Ax
	MV(&val[0],&col[0],&ptr[0],n,&x[0],&Ax[0]);
    for(int i=0;i<n;i++)
        r[i]=b[i]-Ax[i];

	////TODO: d=r
	for(int i=0;i<n;i++)
		d[i]=r[i];

	////TODO: delta_new=rTr
	delta_new=Dot(&r[0],&r[0],n);

	////Here we use the absolute tolerance instead of a relative one, which is slightly different from the notes
	while(iter<max_iter&& delta_new>tol){
        cout<<"entering while"<<endl;
        cout<<delta_new<<endl;
		////TODO: q=Ad
//        MV(val,col,ptr,n,d,Ad);
//        q=Ad;
		MV(&val[0],&col[0],&ptr[0],n,&d[0],&q[0]);

		////TODO: alpha=delta_new/d^Tq
        alpha=delta_new/Dot(&d[0],&q[0],n);

		////TODO: x=x+alpha*d
        for(int i=0; i<n; i++){
            x[i]=x[i]+alpha*d[i];
		}	

		if(iter%50==0&&iter>1){
			////TODO: r=b-Ax
            MV(&val[0],&col[0],&ptr[0],n,&x[0],&Ax[0]);
            for(int i=0;i<n;i++){
                r[i]=b[i]-Ax[i];
            }
		}
		else{
			////TODO: r=r-alpha*q
            for(int i=0;i<n;i++){
                r[i]=r[i]-alpha*q[i];
            }
		}

		////TODO: delta_old=delta_new
		delta_old=delta_new;

		////TODO: delta_new=r^Tr
		delta_new=Dot(&r[0],&r[0],n);

		////TODO: beta=delta_new/delta_old
		beta=delta_new/delta_old;
		
		////TODO: d=r+beta*d
        for(int i=0;i<n;i++){
            d[i]=r[i]+beta*d[i];
		}

		////TODO: increase the counter
		iter++;
	}

	if(iter<max_iter)
		cout<<"CPU conjugate gradient solver converges after "<<iter<<" iterations with residual "<<(delta_new)<<endl;
	else 
		cout<<"CPU conjugate gradient solver does not converge after "<<max_iter<<" iterations with residual "<<(delta_new)<<endl;
}

//////////////////////////////////////////////////////////////////////////
////TODO 3: implement your GPU-based conjugate gradient solver
////Put your CUDA variables and functions here
__global__ void MV_GPU(const double* val,const int* col,const int* ptr,const double* v,double* mv)
{	
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	double element = 0.0;
	
	for (int k = ptr[tid]; k < ptr[tid+1]; k++)
		element = element + val[k]*v[col[k]];

	mv[tid] = element;
}

__global__ void Dot_GPU(double* a,double* b, double* dp)
{
	int tid = threadIdx.x;
	
	__shared__ double dp_dev[grid_size];

	dp_dev[tid]=0;
	__syncthreads();

	for(int k=0;k<grid_size;k++){
		dp_dev[tid] += a[k*grid_size+tid] * b[k*grid_size+tid];
		__syncthreads();
	}
	//printf("%i, %.4f\n",tid, dp_dev[tid]);

	for(unsigned int s=grid_size/2;s>0;s/=2)
	{
		if(tid<s){
			dp_dev[tid] += dp_dev[tid+s];
		}
		__syncthreads();
	}

	*dp = dp_dev[0];
}

__global__ void Add_GPU(double* a, double* b, double* result, double factor)
{
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	
	result[tid] = a[tid]+factor*b[tid];
}
//////////////////////////////////////////////////////////////////////////



ofstream out;

//////////////////////////////////////////////////////////////////////////
////Test functions
////Here we setup a test example by initializing the same Poisson problem as in the last competition: -laplace(p)=b, with p=x^2+y^2 and b=-4.
////The boundary conditions are set on the one-ring ghost cells of the grid
////There is nothing you need to implement in this function

void Initialize_2D_Poisson_Problem(vector<double>& val,vector<int>& col,vector<int>& ptr,vector<double>& b)
{
	////assemble the CRS sparse matrix
	////The grid dimension is grid_size x grid_size. 
	////The matrix's dimension is s x s, with s= grid_size*grid_size.
	////We also initialize the right-hand vector b

	val.clear();
	col.clear();
	ptr.resize(s+1,0);
	b.resize(s,-4.);

	for(int i=0;i<grid_size;i++){
		for(int j=0;j<grid_size;j++){
			int r=I(i,j);
			int nnz_for_row_r=0;

			////set (i,j-1)
			if(!(B(i,j-1))){
				int c=I(i,j-1);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)(i*i+(j-1)*(j-1));	
				b[r]+=boundary_val;
			}

			////set (i-1,j)
			if(!(B(i-1,j))){
				int c=I(i-1,j);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)((i-1)*(i-1)+j*j);
				b[r]+=boundary_val;
			}

			////set (i+1,j)
			if(!(B(i+1,j))){
				int c=I(i+1,j);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)((i+1)*(i+1)+j*j);
				b[r]+=boundary_val;
			}

			////set (i,j+1)
			if(!(B(i,j+1))){
				int c=I(i,j+1);
				val.push_back(-1.);
				col.push_back(c);
				nnz_for_row_r++;
			}
			else{
				double boundary_val=(double)(i*i+(j+1)*(j+1));
				b[r]+=boundary_val;
			}

			////set (i,j)
			{
				val.push_back(4.);
				col.push_back(r);
				nnz_for_row_r++;
			}
			ptr[r+1]=ptr[r]+nnz_for_row_r;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
////CPU test function
////There is nothing you need to implement in this function
void Test_CPU_Solvers()
{
	vector<double> val;
	vector<int> col;
	vector<int> ptr;
	vector<double> b;
	Initialize_2D_Poisson_Problem(val,col,ptr,b);

	vector<double> x(s,0.);
	vector<double> r(s,0.);
	vector<double> q(s,0.);
	vector<double> d(s,0.);
	
	auto start=chrono::system_clock::now();

	Conjugate_Gradient_Solver(&val[0],&col[0],&ptr[0],s,
								&r[0],&q[0],&d[0],
								&x[0],&b[0],
								max_iter_num,tolerance);

	auto end=chrono::system_clock::now();
	chrono::duration<double> t=end-start;
	double cpu_time=t.count()*1000.;	

	if(verbose){
		cout<<"\n\nx for CG on CPU:\n";
		for(int i=0;i<s;i++){
			cout<<x[i]<<", ";
		}	
	}
	cout<<"\n\n";

	//////calculate residual
	MV(&val[0],&col[0],&ptr[0],s,&x[0],&r[0]);
	for(int i=0;i<s;i++)r[i]=b[i]-r[i];
	double residual=Dot(&r[0],&r[0],s);
	cout<<"\nCPU time: "<<cpu_time<<" ms"<<endl;
	cout<<"Residual for your CPU solver: "<<residual<<endl;

	out<<"R0: "<<residual<<endl;
	out<<"T0: "<<cpu_time<<endl;
}

//////////////////////////////////////////////////////////////////////////
////GPU test function
void Test_GPU_Solver()
{
	vector<double> val;
	vector<int> col;
	vector<int> ptr;
	vector<double> b;
	Initialize_2D_Poisson_Problem(val,col,ptr,b);

	vector<double> x(s,0.);
	vector<double> r(s,0.);
	vector<double> q(s,0.);
	vector<double> d(s,0.);


	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//////////////////////////////////////////////////////////////////////////
	////TODO 4: call your GPU functions here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final variables should be stored in the same place as the CPU function, i.e., the array of x
	////The correctness of your simulation will be evaluated by the residual (<1e-3)
	//////////////////////////////////////////////////////////////////////////
	double* val_dev = 0;
	int* col_dev = 0;
	int* ptr_dev = 0;
	double* b_dev = 0;
	double* x_dev = 0;
	double* r_dev = 0;
	double* q_dev = 0;
	double* d_dev = 0;
	double* Ax_dev = 0;

	int iter=0;
	double delta_old=0;
	double* delta_new_dev=0;
	double delta_new=0.0;
	double alpha=0.0;
	double beta=0.0;
	vector<double> Ax(s,0.);
	double* dq_dev=0;
	double dq=0.0;

	cudaMalloc((void**)&val_dev,ptr.back()*sizeof(double));
	cudaMalloc((void**)&col_dev,ptr.back()*sizeof(int));
	cudaMalloc((void**)&ptr_dev,(s+1)*sizeof(int));
	cudaMalloc((void**)&b_dev,s*sizeof(double));
	cudaMalloc((void**)&x_dev,s*sizeof(double));
	cudaMalloc((void**)&r_dev,s*sizeof(double));
	cudaMalloc((void**)&q_dev,s*sizeof(double));
	cudaMalloc((void**)&d_dev,s*sizeof(double));
	cudaMalloc((void**)&Ax_dev,s*sizeof(double));
	cudaMalloc((void**)&delta_new_dev,sizeof(double));
	cudaMalloc((void**)&dq_dev,sizeof(double));

	//cout<<val.size()<<","<<ptr.back()<<endl;
	cudaMemcpy(val_dev,&val[0],ptr.back()*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(col_dev,&col[0],ptr.back()*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(ptr_dev,&ptr[0],(s+1)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev,&b[0],s*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(x_dev,&x[0],s*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(r_dev,&r[0],s*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(q_dev,&q[0],s*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_dev,&d[0],s*sizeof(double),cudaMemcpyHostToDevice);

	MV_GPU<<<grid_size,grid_size>>>(val_dev,col_dev,ptr_dev,x_dev,Ax_dev);

	Add_GPU<<<grid_size,grid_size>>>(b_dev,Ax_dev,r_dev,-1.0);

	Add_GPU<<<grid_size,grid_size>>>(r_dev,d_dev,d_dev,0.0);

	Dot_GPU<<<1,grid_size>>>(r_dev,r_dev,delta_new_dev);
	cudaMemcpy(&delta_new,delta_new_dev,sizeof(double),cudaMemcpyDeviceToHost);

	while(iter<max_iter_num && delta_new>tolerance){

		MV_GPU<<<grid_size,grid_size>>>(val_dev,col_dev,ptr_dev,d_dev,q_dev);

		Dot_GPU<<<1,grid_size>>>(d_dev,q_dev,dq_dev);
		cudaMemcpy(&dq,dq_dev,sizeof(double),cudaMemcpyDeviceToHost);
		alpha=delta_new/dq;

		Add_GPU<<<grid_size,grid_size>>>(x_dev,d_dev,x_dev,alpha);
		cudaMemcpy(&x[0],x_dev,s*sizeof(double),cudaMemcpyDeviceToHost);
		
		if(iter%50==0&&iter>1){
			MV_GPU<<<grid_size,grid_size>>>(val_dev,col_dev,ptr_dev,x_dev,Ax_dev);
			Add_GPU<<<grid_size,grid_size>>>(b_dev,Ax_dev,r_dev,-1.0);
		}
		else{
			Add_GPU<<<grid_size,grid_size>>>(r_dev,q_dev,r_dev,-alpha);
		}

		delta_old=delta_new;

		Dot_GPU<<<1,grid_size>>>(r_dev,r_dev,delta_new_dev);
		cudaMemcpy(&delta_new,delta_new_dev,sizeof(double),cudaMemcpyDeviceToHost);

		beta=delta_new/delta_old;
		
		Add_GPU<<<grid_size,grid_size>>>(r_dev,d_dev,d_dev,beta);

		iter++;
	}
	cudaMemcpy(&x[0],x_dev,sizeof(double),cudaMemcpyDeviceToHost);

	if(iter<max_iter_num)
		cout<<"GPU conjugate gradient solver converges after "<<iter<<" iterations with residual "<<(delta_new)<<endl;
	else 
		cout<<"GPU conjugate gradient solver does not converge after "<<max_iter_num<<" iterations with residual "<<(delta_new)<<endl;

	
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	if(verbose){
		cout<<"\n\nx for CG on GPU:\n";
		for(int i=0;i<s;i++){
			cout<<x[i]<<", ";
		}	
	}
	cout<<"\n\n";

	//////calculate residual
	MV(&val[0],&col[0],&ptr[0],s,&x[0],&r[0]);
	for(int i=0;i<s;i++)r[i]=b[i]-r[i];
	double residual=Dot(&r[0],&r[0],s);
	cout<<"\nGPU time: "<<gpu_time<<" ms"<<endl;
	cout<<"Residual for your GPU solver: "<<residual<<endl;

	out<<"R1: "<<residual<<endl;
	out<<"T1: "<<gpu_time<<endl;
}

int main()
{
	if(name::team=="Team_X"){
		printf("\nPlease specify your team name and team member names in name::team and name::author to start.\n");
		return 0;
	}

	std::string file_name=name::team+"_competition_final_conjugate_gradient.dat";
	out.open(file_name.c_str());
	
	if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}

	Test_CPU_Solvers();
	Test_GPU_Solver();

	return 0;
}