#include <stdio.h>
#include <stdlib.h>
#include "sha1.h"
#include "sha2.h"

#define MAX_MEM 2048

__device__ const unsigned char *m = "Goodbye, cruel world!";

__global__ void kernel_sha1(unsigned char *hval) {
  sha1_ctx ctx[1];

  sha1_begin(ctx);
  sha1_hash(m, 21UL, ctx);
  sha1_end(hval, ctx);
}

__global__ void kernel_sha2(unsigned char *hval) {
  sha256_ctx ctx[1];

  sha256_begin(ctx);
  sha256_hash(m, 21UL, ctx);
  sha256_end(hval, ctx);
}

int main( void ) {
  unsigned int i;
  cudaError_t err;
  unsigned char *host_hval;
  unsigned char *device_hval;

  /* allocate memory */
  host_hval = (unsigned char *)malloc(MAX_MEM);
  if(host_hval == NULL) {}
  err = cudaMalloc((void**)&device_hval, MAX_MEM);
  if(err != cudaSuccess) {}

  /* test SHA1 */
  printf("Running SHA1 kernel: ");
  kernel_sha1<<<1,1>>>(device_hval);
  err = cudaMemcpy(host_hval, device_hval, SHA1_DIGEST_SIZE, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {}

  /* should print: f99ec870820d0832c9e04e9e4a0e40fb5038636c */
  for(i=0; i<SHA1_DIGEST_SIZE; i++)
    printf("%02X", host_hval[i]);
  printf("\n");

  /* test SHA2 */
  printf("Running SHA2 kernel: ");
  kernel_sha2<<<1,1>>>(device_hval);
  err = cudaMemcpy(host_hval, device_hval, SHA256_DIGEST_SIZE, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {}

  /* should print: 469c24f94970733aa9d3c18ba88b816a5572cdc86286c30107e3ffcb9ef88e05 */
  for(i=0; i<SHA256_DIGEST_SIZE; i++)
    printf("%02X", host_hval[i]);
  printf("\n");

  /* clean up */
  err = cudaFree(device_hval);
  if(err != cudaSuccess) {}
  device_hval = NULL;
  free(host_hval);
  host_hval = NULL;

  printf( "Exiting...\n" );
  return 0;
}
