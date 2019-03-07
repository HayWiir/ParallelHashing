//mpicc parallel_hashing.c -o sd -lssl -lcrypto
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <openssl/md5.h>

#define BLOCK_SIZE 512	 //Block size in bytes -> 512, smaller values for testing

int mergeandhash(int *hashed_blocks, int index)
{
	//TODO
	//Implement fake padding


}

int main(int argc, char** argv)
{	

	//printf("Hello\n");
	int size, rank, i;
	long int file_size, offset, num_blocks, num_blocks_per_process, hash_count;
	char* curr;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//printf("Creating\n");
		
	FILE* fp = fopen("test_file.txt", "r"); //test_file.txt is the main input file, test.txt for testing
	if(fp==NULL)
	{
		printf("No such file exists\n");
		exit(0);
	}

	if(rank==0)
	{
		fseek(fp, 0L, SEEK_END);
		file_size = ftell(fp);
		rewind(fp);

		num_blocks = file_size/BLOCK_SIZE;
		num_blocks_per_process = num_blocks/size + 1;
	}

	MPI_Bcast(&file_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_blocks, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_blocks_per_process, 1, MPI_LONG, 0, MPI_COMM_WORLD);

	hash_count=0;
	int **hashed_blocks = (int**) malloc(num_blocks_per_process * sizeof(int*));
	for(i=0;i<num_blocks_per_process; i++)
	{
		hashed_blocks[i] = (int *) malloc(MD5_DIGEST_LENGTH * sizeof(int));
	}
	

	offset = rank*BLOCK_SIZE; //Get first offset for each process
	fseek(fp, offset, SEEK_SET);

	size_t result;
	curr = (char*) malloc((BLOCK_SIZE)*sizeof(char));	

	//Initial Block hashing
	while((result = fread(curr, sizeof(char), BLOCK_SIZE, fp)) ==  sizeof(char)*BLOCK_SIZE)
	{	
		offset += size*BLOCK_SIZE;
		fseek(fp, offset, SEEK_SET);

		unsigned char RES[MD5_DIGEST_LENGTH];         //Hashing individual blocks
 		MD5(curr, sizeof(curr)/sizeof(curr[0]), RES);

		for(i=0; i<MD5_DIGEST_LENGTH; i++)
	 		hashed_blocks[hash_count][i] = RES[i];
	 	hash_count++;
		
		// if(rank == 4)
		// {	
		// 	printf("Process%d ", rank);
		// 	for(i = 0; i < MD5_DIGEST_LENGTH; i++)
  //   			printf("%02x", RES[i]);
		// 	printf("\tNext offset:%ld\n", offset);//test only

		// }
	}

	//Padding unecessary for current hashing algo (MD5) (For blocks with size<BLOCK_SIZE)
	if(result<BLOCK_SIZE && result>0){
		unsigned char RES[MD5_DIGEST_LENGTH];        
 		MD5(curr, sizeof(curr)/sizeof(curr[0]), RES);

 		for(i=0; i<MD5_DIGEST_LENGTH; i++)
	 		hashed_blocks[hash_count][i] = RES[i];
	 	hash_count++;
	}	


	
	MPI_Finalize();

	return 0;
}	