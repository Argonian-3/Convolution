#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/mat.hpp"
#include<sys/time.h>
#include<stdio.h>
#include<stdlib.h>
#include<inttypes.h>
#include<float.h>
using namespace cv;
using namespace std;
const static int FILTER_RADIUS=2;
const static int FILTER_WIDTH=FILTER_RADIUS*2+1;
const static int MAX_BLOCK_SIZE=32;
const static int OUTER_TILE_WIDTH=MAX_BLOCK_SIZE;
const static int INNER_TILE_WIDTH=OUTER_TILE_WIDTH-2*FILTER_RADIUS;
const static int FILTER_SQUARE=FILTER_WIDTH*FILTER_WIDTH;
__constant__ unsigned int D_FILTER_WIDTH;
__constant__ unsigned int D_FILTER_RADIUS;
__constant__ unsigned int D_FILTER_SQUARE;
__constant__ unsigned int D_OUTER_TILE_WIDTH;
void check(cudaError_t cudaRet) {
	if (cudaRet!=cudaSuccess) {
		printf("Error: %s:%d, ",__FILE__,__LINE__);
		printf("code: %d, reason:%s\n",cudaRet,cudaGetErrorString(cudaRet));
		exit(-1);
	}
}
double getTime() {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec+(double)tp.tv_usec/1.0e6);
}
bool equal(const float x,const float y) {
	const float relativeTolerance=1e-2;
	if (fabs(x-y) < relativeTolerance)
		return true;
	return false;
}
bool areEqual(const unsigned char *a,const unsigned char *b,const unsigned int rows,const unsigned int cols) {
	for (unsigned int row=0;row<rows;row++) {
		for (unsigned int col=0;col<cols;col++) {
			if (!equal((float)a[row*cols+col]/255,(float)b[row*cols+col]/255)) {
				return false;
			}
		}
	}
	return true;
}
bool equal(const Mat a,const Mat b) {
	if (a.rows!=b.rows||a.cols!=b.cols||a.size!=b.size||a.channels()!=b.channels())
		return false;
	for (int row=0;row<a.rows;row++) {
		const unsigned char* aRow=a.ptr<unsigned char>(row);
		const unsigned char* bRow=b.ptr<unsigned char>(row);
		for (int col=0;col<a.cols*a.channels();col++) {
			if (!equal((float)a.at<unsigned char>(row,col)/255,(float)b.at<unsigned char>(row,col)/255))
				return false;
		}
	}
	return true;
}
void printColoredMat(const Mat m) {
	if (m.rows>10)
		return;
	for (int row=0;row<m.rows;row++) {
		for (int col=0;col<m.cols;col++) {
			Vec3b s = m.at<Vec3b>(row,col);
			printf("(%u,%u,%u),",s.val[0],s.val[1],s.val[2]);
		}
		printf("\n");
	}
}
void printGrayMat(const Mat m) {
	if (m.rows>10)
		return;
	for (int row=0;row<m.rows;row++) {
		for (int col=0;col<m.cols;col++) {
			printf("%u,",m.at<unsigned char>(row,col));
		}
		printf("\n");
	}
}
Mat cpuGrayBlur(const Mat input) {
	if (input.dims>2) {
		printf("Input matrix of dimension %i is greater than 2",input.dims);
		exit(-1);
	}
	const int rows=input.rows;
	const int cols=input.cols;
	if (rows<FILTER_WIDTH||cols<FILTER_WIDTH) {
		printf("Image is too small for filter width %i\n",FILTER_WIDTH);
		exit(-1);
	}
	CV_Assert(input.depth()==CV_8U);
	Mat result=Mat::zeros(input.size(),input.type());
	for (unsigned int row=0;row<rows;row++) {
		for (unsigned int col=0;col<cols;col++) {
			unsigned int sum=0;
			for (int i=-FILTER_RADIUS;i<=FILTER_RADIUS;i++) {
				for (int j=-FILTER_RADIUS;j<=FILTER_RADIUS;j++) {
					if (row+i<rows&&col+j<cols) {
						sum=sum+ input.at<unsigned char>(row+i,col+j);
					}/* 
					else if (row+i>=rows&&col+j>=cols) {
						sum=sum+input.at<unsigned char>(row-i,col-j);
					}
					else if (row+i>=rows) {
						sum=sum+input.at<unsigned char>(row-i,col+j);
					}
					else {
						sum=sum+input.at<unsigned char>(row+i,col-j);
					} */
				}
			}
			result.at<unsigned char>(row,col)=(unsigned char)(sum/FILTER_SQUARE);
		}
	}
	return result;
}
__global__ void gpuGrayBlur(unsigned char *out,const unsigned char *input,const int rows,const int cols) {
	unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
	if (rows>=FILTER_WIDTH&&cols>=FILTER_WIDTH&&row<rows&&col<cols) {
		unsigned int sum=0;
		for (int i=-FILTER_RADIUS;i<=FILTER_RADIUS;i++) {
			for (int j=-FILTER_RADIUS;j<=FILTER_RADIUS;j++) {
				if (row+i<rows&&col+j<cols) {
					sum=sum+input[cols*(row+i)+col+j];
				}
				/* else if (row+i>=rows&&col+j>=cols) {
					sum=sum+input[cols*(row-i)+col-j];
				}
				else if (row+i>=rows) {
					sum=sum+input[cols*(row-i)+col+j];
				}
				else {
					sum=sum+input[cols*(row+i)+col-j];
				} */
			}
		}
		out[row*cols+col]=(unsigned char)(sum/FILTER_SQUARE);
	}
}	
__global__ void gpuTiledGrayBlur(unsigned char *out,const unsigned char *input,const int rows,const int cols) {
	const unsigned int col = blockIdx.x*INNER_TILE_WIDTH+threadIdx.x-FILTER_RADIUS;
	const unsigned int row = blockIdx.y*INNER_TILE_WIDTH+threadIdx.y-FILTER_RADIUS;
	__shared__ unsigned char outerTile[OUTER_TILE_WIDTH][OUTER_TILE_WIDTH];
	if (row<rows&&col<cols) {
		outerTile[threadIdx.y][threadIdx.x]=input[row*cols+col];
	}
	else
		outerTile[threadIdx.y][threadIdx.x]=0;
	__syncthreads();
	const int tileCol=threadIdx.x-FILTER_RADIUS;
	const int tileRow=threadIdx.y-FILTER_RADIUS;
	if (col<cols&&row<rows) {
		if (tileCol<INNER_TILE_WIDTH&&tileRow<INNER_TILE_WIDTH&&tileCol>=0&&tileRow>=0) {
			unsigned int sum=0;
			for (int i=0;i<=2*FILTER_RADIUS;i++) {
				for (int j=0;j<=2*FILTER_RADIUS;j++) {
					sum=sum+outerTile[tileRow+i][tileCol+j];
				}
			}
			out[row*cols+col]=(unsigned char)(sum/FILTER_SQUARE);
		}
	}
}
int main(int argCount, char* argv[]) {
	if (MAX_BLOCK_SIZE>32) {
		printf("Block size is too large for tiles!\n");
		exit(-1);
	}
	cudaDeviceSynchronize();
	double start,end;
	if (argCount<2) {
		printf("No input file given.\n");
		exit(-1);
	}
	if (OUTER_TILE_WIDTH-2*FILTER_RADIUS<=0) {
		printf("Filter radius is too large!\n");
		exit(-1);
	}
	// imread default returns pixels in BGR order:need subpixel access for at
	// grayscale read: each at is one grayscale pixel
	const Mat reference = imread(argv[1],IMREAD_GRAYSCALE);
	if (reference.empty()) {
		printf("Could not read image: %s\n",argv[1]);
		exit(-1);
	}
	if (reference.rows<FILTER_WIDTH||reference.cols<FILTER_WIDTH) {
		printf("Either the picture is too small or the filter width is too large!\n");
		exit(-1);
	}
	Mat cvBlur;
	Size *fiveSquare=new Size_(FILTER_WIDTH,FILTER_WIDTH);
	Point *anchor = new Point_(-1,-1);
	start=getTime();
	blur(reference,cvBlur,*fiveSquare,*anchor,BORDER_CONSTANT);
	end=getTime();
	printf("OpenCV: %fs\n",end-start);
	start=getTime();
	Mat cpu=cpuGrayBlur(reference);
	end=getTime();
	printf("CPU: %fs\n",end-start);
	printf("Correct result:%i\n",equal(cvBlur,cpu));
	
	unsigned char* inDevice;
	unsigned char* outDevice;
	check(cudaMalloc((void**)&inDevice,reference.rows*reference.cols*sizeof(unsigned char)));
	check(cudaMalloc((void**)&outDevice,reference.rows*reference.cols*sizeof(unsigned char)));
	cudaMemcpy(inDevice,reference.ptr<unsigned char>(),reference.rows*reference.cols*sizeof(unsigned char),cudaMemcpyHostToDevice);
	const Mat zeros = Mat::zeros(reference.size(),reference.type());
	cudaMemcpy(outDevice,zeros.ptr<unsigned char>(),reference.rows*reference.cols*sizeof(unsigned char),cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(D_FILTER_RADIUS,&FILTER_RADIUS,sizeof(int));
	cudaMemcpyToSymbol(D_FILTER_SQUARE,&FILTER_SQUARE,sizeof(int));
	cudaMemcpyToSymbol(D_FILTER_WIDTH,&FILTER_WIDTH,sizeof(int));
	start=getTime();
	unsigned char *outHost=(unsigned char*)malloc(reference.rows*reference.cols*sizeof(unsigned char));
	{
	dim3 blockDim(MAX_BLOCK_SIZE,MAX_BLOCK_SIZE);
	dim3 gridDim(16*ceil(reference.rows/MAX_BLOCK_SIZE),16*ceil(reference.cols/MAX_BLOCK_SIZE));
	gpuGrayBlur<<<gridDim,blockDim>>>(outDevice,inDevice,reference.rows,reference.cols);
	}
	check(cudaDeviceSynchronize());
	cudaMemcpy(outHost,outDevice,reference.rows*reference.cols*sizeof(unsigned char),cudaMemcpyDeviceToHost);
	end=getTime();
	printf("Simple GPU: %fs\n",end-start);
	printf("Correct result:%i\n",areEqual(&cvBlur.at<unsigned char>(),outHost,cvBlur.rows,cvBlur.cols));
	Mat simpleGPUMat(reference.size(),reference.type());
	for (int row=0;row<reference.rows;row++) {
		for (int col=0;col<reference.cols;col++) {
			simpleGPUMat.at<unsigned char>(row,col)=outHost[row*reference.cols+col];
		}
	}
	cudaMemcpy(outDevice,zeros.ptr<unsigned char>(),reference.rows*reference.cols*sizeof(unsigned char),cudaMemcpyHostToDevice);
	start=getTime();
	{
	dim3 blockDim(MAX_BLOCK_SIZE,MAX_BLOCK_SIZE);
	dim3 gridDim((reference.cols+INNER_TILE_WIDTH-1)/INNER_TILE_WIDTH,(reference.rows+INNER_TILE_WIDTH-1)/INNER_TILE_WIDTH);
	const unsigned int numSharedBytesPerBlock=OUTER_TILE_WIDTH*OUTER_TILE_WIDTH*sizeof(unsigned char);
	gpuTiledGrayBlur<<<gridDim,blockDim,numSharedBytesPerBlock>>>(outDevice,inDevice,reference.rows,reference.cols);
	}
	check(cudaDeviceSynchronize());
	cudaMemcpy(outHost,outDevice,reference.rows*reference.cols*sizeof(unsigned char),cudaMemcpyDeviceToHost);
	end=getTime();
	printf("Tiled GPU: %fs\n",end-start);
	printf("Correct result:%i\n",areEqual(&cvBlur.at<unsigned char>(),outHost,cvBlur.rows,cvBlur.cols));
	Mat tiledGPUMat(reference.size(),reference.type());
	for (int row=0;row<reference.rows;row++) {
		for (int col=0;col<reference.cols;col++) {
			tiledGPUMat.at<unsigned char>(row,col)=outHost[row*reference.cols+col];
		}
	}
	imwrite("blurredImg_opencv.jpg",cvBlur);
	imwrite("blurredImg_cpu.jpg",cpu);
	imwrite("blurredImg_gpu.jpg",simpleGPUMat);
	imwrite("blurredImg_tiled_gpu.jpg",tiledGPUMat);
	exit(0);
}