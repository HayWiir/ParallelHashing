#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define BLOCK_SIZE 8 //Block size in bytes -> 512

int main(int argc, char** argv)
{	

	//printf("Hello\n");
	int size, rank, i;
	long int file_size, offset;
	char* curr;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//printf("Creating\n");
		
	FILE* fp = fopen("test.txt", "r"); //test_file.txt is input file
	if(fp==NULL)
	{
		printf("No such file exists\n");
		exit(0);
	}

	curr = (char*) malloc((BLOCK_SIZE+1)*sizeof(char)); //REMOVE EXTRA CHAR
	curr[BLOCK_SIZE]=0;//REMOVE 

	
	offset = rank*BLOCK_SIZE;
	fseek(fp, offset, SEEK_SET);

	size_t result;

	
	while(result = fread(curr, sizeof(char), BLOCK_SIZE, fp) ==  sizeof(char)*BLOCK_SIZE)
	{	
		offset += size*BLOCK_SIZE;
		fseek(fp, offset, SEEK_SET);

		// TODO
		// Hashing individual blocks
		
		if(rank == 4)printf("Process%d: %s Next offset:%ld\n",rank, curr, offset);
		
	}

	//Add padding
	if(result<BLOCK_SIZE && result>0){
		printf("%ld\n",result );
	}


	
	MPI_Finalize();

	return 0;
}	