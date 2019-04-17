#include "md5_cuda.h"
#define BLOCK_SIZE 512
#define NTHREAD 1024

__global__ void hash_blocks_intra(char* data, unsigned char* hash_table, int* d_filesize)
{
	int i, j, k, curr_index, merge_index;
	int id = threadIdx.x;
	int num_blocks = *d_filesize/BLOCK_SIZE;
	int blocks_per_thread = num_blocks/blockDim.x;
	int data_offset = id*BLOCK_SIZE*blocks_per_thread;
	int hash_table_offset = id*MD5_DIGEST_LENGTH;
	char* curr;
	unsigned char odd_hash[MD5_DIGEST_LENGTH], even_hash[MD5_DIGEST_LENGTH], final_hash[MD5_DIGEST_LENGTH], merged_hashes[2*MD5_DIGEST_LENGTH];
	/*
	 * Go through all the blocks alloted to the thread.
	 * Hash each block.
	 * When 2 consecutive blocks are hashed, merge them and get their hash.
	 * Merge this merged_hash with the final_hash and store result in final_hash
	 */
	//if(id==3) printf("%d %d %d %d", num_blocks, blocks_per_thread, data_offset, hash_table_offset);
	curr = (char*) malloc(BLOCK_SIZE*sizeof(char));
	for(i=0; i<blocks_per_thread; i++)
	{
		//if(id==3)printf("Hi %d\n", i);

		curr_index=0;
		for(j=0; j<BLOCK_SIZE; j++)
		{
		//	if(id==3) printf("%ld  %d\n", data_offset + i*BLOCK_SIZE + j, i);
			curr[curr_index++] = data[data_offset + i*BLOCK_SIZE + j];
		}

		(i%2==0)?md5((uint*) curr, (uint*) even_hash):md5((uint*) curr, (uint*) odd_hash);

		if(i%2)
		{
			merge_index=0;
			for(k=0; k<MD5_DIGEST_LENGTH; k++)
				merged_hashes[merge_index++] = even_hash[k];
			for(k=0; k<MD5_DIGEST_LENGTH; k++)
				merged_hashes[merge_index++] = odd_hash[k];

			//store merged hash in even_hash
			md5((uint*) merged_hashes, (uint*) even_hash);

			if(i==1)
			{
				for(k=0; k<MD5_DIGEST_LENGTH; k++)
					final_hash[k] = even_hash[k];
			}
			else
			{
				merge_index=0;
				for(k=0; k<MD5_DIGEST_LENGTH; k++)
					merged_hashes[merge_index++] = final_hash[k];
				for(k=0; k<MD5_DIGEST_LENGTH; k++)
					merged_hashes[merge_index++] = even_hash[k];

				md5((uint*) merged_hashes, (uint*) final_hash);

			}

		}

//		if(id==255 && i>1)
//			{
//				printf("%d T%d Hi ", i, id);
//				for(k=0; k<MD5_DIGEST_LENGTH; k++)
//						printf("%02x", (unsigned char) final_hash[k]);
//					printf("\n");
//			}

	}


//	printf("Thread %d: ", id);
//	for(i=0; i<MD5_DIGEST_LENGTH; i++)
//		printf("%02x", (unsigned char) final_hash[i]);
//	printf("\n");


	for(k=0;k<MD5_DIGEST_LENGTH; k++)
	{
		hash_table[hash_table_offset + k] = final_hash[k];
	}


}


__global__ void hash_blocks_inter(unsigned char* hash_table)
{
	int i, k, other_thread, merge_index;
	int id = threadIdx.x;
	int hash_table_offset = id*MD5_DIGEST_LENGTH;
	unsigned char merged_hashes[2*MD5_DIGEST_LENGTH], res_hash[MD5_DIGEST_LENGTH];

	int nthread = NTHREAD;
	int jump = 2;
	while(nthread!=1)
	{
		nthread = nthread>>1;
		for(i=0; i<NTHREAD; i+=(jump/2))
		{
			if(id==i && i%jump==0)
			{
				other_thread = id + (jump/2);
				__syncthreads();
				merge_index=0;
				for(k=0; k<MD5_DIGEST_LENGTH; k++)
					merged_hashes[merge_index++] = hash_table[hash_table_offset + k];
				for(k=0; k<MD5_DIGEST_LENGTH; k++)
					merged_hashes[merge_index++] = hash_table[other_thread*MD5_DIGEST_LENGTH + k];


				md5((uint*) merged_hashes, (uint*) res_hash);

				for(k=0;k<MD5_DIGEST_LENGTH; k++)
				{
					hash_table[hash_table_offset + k] = res_hash[k];
				}

			}
		}
		jump*=2;
		__syncthreads();
	}

//	if(id==0)
//	{
//		printf("Thread 0: ");
//		for(i=0; i<MD5_DIGEST_LENGTH; i++)
//			printf("%02x", (unsigned char) hash_table[i]);
//		printf("\n");
//	}
//	__syncthreads();
}
