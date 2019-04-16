//mpicc parallel_hashing.c -o sd -lssl -lcrypto
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <openssl/md5.h>

#define BLOCK_SIZE 512	 //Block size in bytes -> 512, smaller values for testing

//TODO
//Take file name as command line arg
//Assert num processes to be power of 2

void mergeandhash(unsigned char **hashed_blocks, int first_index)
{
	int i, j, n;
	unsigned char RES[MD5_DIGEST_LENGTH];   
	unsigned char new[2*MD5_DIGEST_LENGTH];

	for(i=first_index*2, n=0; n<2 ; i++, n++)
	{
		for(j=0; j<MD5_DIGEST_LENGTH; j++)
		{
			new[n*MD5_DIGEST_LENGTH + j] = hashed_blocks[i][j];
		}
	}

 	//Below func call results in different hash ??
 	//MD5(hashed_blocks[first_index*2], 2 * (sizeof(hashed_blocks[first_index*2])/sizeof(hashed_blocks[first_index*2][0])), RES);
	
	MD5(new, sizeof(new), RES);
	for(i=0; i<MD5_DIGEST_LENGTH; i++)
	{
		hashed_blocks[first_index][i] = RES[i];
	}

}



int main(int argc, char** argv)
{	

	//printf("Hello\n");
	int size, rank, i, tree_height, jump;
	long int file_size, offset, num_blocks, num_blocks_per_process, hash_count;
	char* curr;
	MPI_Status status;

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
	unsigned char **hashed_blocks = (unsigned char**) malloc(num_blocks_per_process * sizeof(unsigned char*));
	for(i=0;i<num_blocks_per_process; i++)
	{
		hashed_blocks[i] = (unsigned char *) malloc(MD5_DIGEST_LENGTH * sizeof(unsigned char));
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
  //   			printf("%02x", hashed_blocks[hash_count-1][i]);
		// 	printf("\tNext offset:%ld\n", offset);//test only

		// }
	}

	//Padding unecessary for current hashing algo (MD5) (For blocks with size<BLOCK_SIZE)
	if(result<BLOCK_SIZE && result>0)
	{
		unsigned char RES[MD5_DIGEST_LENGTH];        
 		MD5(curr, sizeof(curr)/sizeof(curr[0]), RES);

 		for(i=0; i<MD5_DIGEST_LENGTH; i++)
	 		hashed_blocks[hash_count][i] = RES[i];
	 	hash_count++;
	}

	//Merge and hash INTRA-process
	while (hash_count!=1)
	{
		for(i=0; i<hash_count/2; i++)
			mergeandhash(hashed_blocks, i);

		hash_count = hash_count/2 + hash_count%2;
	}

	// printf("Process%d ", rank);
	// for(i = 0; i < MD5_DIGEST_LENGTH; i++)
	// 	printf("%02x", hashed_blocks[0][i]);
	// printf("\n");

	MPI_Barrier(MPI_COMM_WORLD);

	//Implement for pow(2,n) number of processes ONLY

	tree_height = size;
	jump = 2;
	while(tree_height!=1)
	{	
		//size >>= 1;
		tree_height/=2;
		for(i=0; i<size; i+=(jump/2))
		{
			if(rank==i && i%jump==0)
			{	
				//get hash from #rank+jump/2 process
				MPI_Recv(hashed_blocks[1], MD5_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, rank+(jump/2), 0, MPI_COMM_WORLD, &status);


					// printf("Height %d, Process%d ", tree_height, rank);
					// for(i = 0; i < MD5_DIGEST_LENGTH; i++)
					// 	printf("%02x", hashed_blocks[0][i]);
					

					// printf(" Height %d, Process%d ", tree_height, rank+(jump/2));
					// for(i = 0; i < MD5_DIGEST_LENGTH; i++)
					// 	printf("%02x", hashed_blocks[1][i]);
					// printf("\n");



				//merge and hash for process #rank and #rank+(jump/2)
				mergeandhash(hashed_blocks, 0);
			}

			else if(rank==i && i%jump!=0)
			{
				//Send to #rank-(jump/2) process
				MPI_Send(hashed_blocks[0], MD5_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, rank-(jump/2), 0, MPI_COMM_WORLD);
			}			
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		jump*=2;
		//tree_height++;
	}

	//Process 0 has final hash
	if(rank==0)	
	{
		printf("Hash: ");
		for(i = 0; i < MD5_DIGEST_LENGTH; i++)
			printf("%02x", hashed_blocks[0][i]);
		printf("\n");
	}

	MPI_Finalize();

	return 0;
}	