#include <stdio.h>
#include "sha1.h"

__global__ void kernel(unsigned char *hval) {
  sha1_ctx ctx[1];
  const unsigned char *m = "Goodbye, cruel world!";

  sha1_begin(ctx);
  sha1_hash(m, 21UL, ctx);
  sha1_end(hval, ctx);
}

int main( void ) {
  unsigned int i;
  cudaError_t err;
  unsigned char *host_hval;
  unsigned char *device_hval;

  host_hval = (unsigned char *)malloc(SHA1_DIGEST_SIZE);
  if(host_hval == NULL) {}
  err = cudaMalloc( (void**)&device_hval, SHA1_DIGEST_SIZE );
  if(err != cudaSuccess) {}

  printf("Running SHA1 kernel...\n");
  kernel<<<1,1>>>(device_hval);

  err = cudaMemcpy(host_hval, device_hval, SHA1_DIGEST_SIZE, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {}
  err = cudaFree(device_hval);
  if(err != cudaSuccess) {}
  device_hval = NULL;

  /* should print: f99ec870820d0832c9e04e9e4a0e40fb5038636c */
  for(i=0; i<SHA1_DIGEST_SIZE; i++)
    printf("%02X", host_hval[i]);
  printf("\n");
  free(host_hval);
  host_hval = NULL;

  printf( "Exiting...\n" );
  return 0;
}
