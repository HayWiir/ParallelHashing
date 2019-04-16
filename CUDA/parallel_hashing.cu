#include <stdio.h>
#include <stdlib.h>
#include "util.h"

#define MAX_FILE_SIZE 5368709120


// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}



int main()
{
	//host data
	int i;
	char* data;
	unsigned char *hash_table, *final_hash;
	float elapsed1=0, elapsed2 = 0;

	//device data
	char* d_data;
	unsigned char* d_hash;
	int* d_filesize;

	//time measurement
	cudaEvent_t start, stop;
	CudaSafeCall(cudaEventCreate(&start));
	CudaSafeCall(cudaEventCreate(&stop));


	FILE* fp = fopen("test_file.txt", "r");
	if(fp==NULL)
	{
		printf("No such file exists\n");
		exit(0);
	}
	size_t filesize;
	data = (char*) malloc(sizeof(char) * MAX_FILE_SIZE);
	filesize = fread(data, sizeof(char), MAX_FILE_SIZE/sizeof(char), fp);



	CudaSafeCall(cudaMalloc((void**) &d_data, filesize));
	CudaSafeCall(cudaMalloc((void**) &d_hash, NTHREAD*MD5_DIGEST_LENGTH*sizeof(unsigned char)));
	CudaSafeCall(cudaMalloc((void**) &d_filesize, sizeof(int)));
	CudaSafeCall(cudaMemcpy(d_data, data, filesize, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_filesize, &filesize, sizeof(int), cudaMemcpyHostToDevice));

	CudaSafeCall(cudaEventRecord(start, 0));
	hash_blocks_intra<<<1,NTHREAD>>>(d_data, d_hash, d_filesize);
	CudaCheckError();
	CudaSafeCall(cudaEventRecord(stop, 0));
	CudaSafeCall(cudaEventSynchronize (stop) );
	CudaSafeCall(cudaEventElapsedTime(&elapsed1, start, stop) );

	hash_table = (unsigned char*) malloc(NTHREAD*MD5_DIGEST_LENGTH*sizeof(unsigned char));
	CudaSafeCall(cudaMemcpy(hash_table, d_hash, NTHREAD*MD5_DIGEST_LENGTH*sizeof(unsigned char), cudaMemcpyDeviceToHost));


	CudaSafeCall(cudaEventRecord(start, 0));
	hash_blocks_inter<<<1,1>>>(d_hash);
	CudaCheckError();
	CudaSafeCall(cudaEventRecord(stop, 0));
	CudaSafeCall(cudaEventSynchronize (stop) );
	CudaSafeCall(cudaEventElapsedTime(&elapsed2, start, stop) );

	final_hash = (unsigned char*) malloc(MD5_DIGEST_LENGTH*sizeof(unsigned char));
	CudaSafeCall(cudaMemcpy(hash_table, d_hash, NTHREAD*MD5_DIGEST_LENGTH*sizeof(unsigned char), cudaMemcpyDeviceToHost));

	printf("Final: ");
	for(i=0; i<MD5_DIGEST_LENGTH; i++)
		printf("%02x", (unsigned char) hash_table[i]);
	printf("\n");
	printf("Time taken by %d threads is %f", NTHREAD, elapsed1+elapsed2);
	cudaFree(d_data);
	cudaFree(d_hash);
	return 0;
}

