#include <stdio.h>
#include <stdlib.h>
#include "sha1.h"
#include "sha2.h"

#define N 37426 /* the (arbitrary) number of hashes we want to calculate */
#define THREAD_COUNT 128

__device__ const unsigned char *m = "Goodbye, cruel world!";

__global__ void kernel_sha1(unsigned char *hval) {
  sha1_ctx ctx[1];
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while(tid < N) {
    sha1_begin(ctx);
    sha1_hash(m, 21UL, ctx);
    sha1_end(hval+tid*SHA1_DIGEST_SIZE, ctx);
    tid += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_sha2(unsigned char *hval) {
  sha256_ctx ctx[1];
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while(tid < N) {
    sha256_begin(ctx);
    sha256_hash(m, 21UL, ctx);
    sha256_end(hval+tid*SHA256_DIGEST_SIZE, ctx);
    tid += blockDim.x * gridDim.x;
  }
}

int main( void ) {
  int device_count;
  unsigned int i, j, blocks, offset;
  float sha1_elapsed, sha256_elapsed;
  cudaError_t err;
  cudaEvent_t start, stop;
  cudaDeviceProp prop;
  unsigned char *host_hval;
  unsigned char *device_hval;

  err = cudaEventCreate(&start);
  if(err != cudaSuccess) { }
  err = cudaEventCreate(&stop);
  if(err != cudaSuccess) { }

  err = cudaGetDeviceCount(&device_count);
  if(err != cudaSuccess) { }
  if(device_count == 0) {
    printf("Device not found.\n");
    return 1;
  }
  // hard-code device 0 for now
  err = cudaGetDeviceProperties(&prop, 0);
  if(err != cudaSuccess) {
    printf("Failed to get device properties.\n");
    return 1;
  }
  blocks = prop.multiProcessorCount * 6;

  /* allocate enough memory for the largest output required */
  host_hval = (unsigned char *)malloc(N*SHA256_DIGEST_SIZE);
  if(host_hval == NULL) {
    printf("Failed to allocate host memory.\n");
    return 1;
  }
  err = cudaMalloc((void**)&device_hval, N*SHA256_DIGEST_SIZE);
  if(err != cudaSuccess) {
    printf("Failed to allocate device memory.\n");
    err = cudaFree(device_hval);
    if(err != cudaSuccess) {
      printf("Failed to free device memory.\n");
    }
    free(host_hval);
    return 1;
  }

  /* test SHA1 */
  err = cudaMemset(device_hval, 0, N*SHA256_DIGEST_SIZE);
  if(err != cudaSuccess) {
    printf("SHA1: Failed to initialize device memory.\n");
    err = cudaFree(device_hval);
    if(err != cudaSuccess) { 
      printf("SHA1: Failed to free device memory.\n");
    }
    free(host_hval);
    return 1;
  }
  err = cudaEventRecord(start, 0);
  if(err != cudaSuccess) { }
  kernel_sha1<<<blocks,THREAD_COUNT>>>(device_hval);
  err = cudaEventRecord(stop, 0);
  if(err != cudaSuccess) { }
  err = cudaEventSynchronize(stop);
  if(err != cudaSuccess) { }
  err = cudaEventElapsedTime(&sha1_elapsed, start, stop);
  if(err != cudaSuccess) { }
  err = cudaMemcpy(host_hval, device_hval, N*SHA1_DIGEST_SIZE, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    printf("SHA1: cudaMemcpy failed.\n");
    err = cudaFree(device_hval);
    if(err != cudaSuccess) {
      printf("SHA1: Failed to free device memory.\n");
    }
    free(host_hval);
    return 1;
  }
  /* should print: f99ec870820d0832c9e04e9e4a0e40fb5038636c */
  for(i=0; i<N; i++) {
    offset = i * SHA1_DIGEST_SIZE;
    for(j=0; j<SHA1_DIGEST_SIZE; j++)
      printf("%02X", host_hval[offset+j]);
    printf("\n");
  }

  /* test SHA2 */
  err = cudaMemset(device_hval, 0, N*SHA256_DIGEST_SIZE);
  if(err != cudaSuccess) {
    printf("SHA256: Failed to initialize device memory.\n");
    err = cudaFree(device_hval);
    if(err != cudaSuccess) {
      printf("SHA256: Failed to free device memory.\n");
    }
    free(host_hval);
    return 1;
  }
  err = cudaEventRecord(start, 0);
  if(err != cudaSuccess) { }
  kernel_sha2<<<blocks,THREAD_COUNT>>>(device_hval);
  cudaEventRecord(stop, 0);
  if(err != cudaSuccess) { }
  err = cudaEventSynchronize(stop);
  if(err != cudaSuccess) { }
  err = cudaEventElapsedTime(&sha256_elapsed, start, stop);
  if(err != cudaSuccess) { }
  err = cudaMemcpy(host_hval, device_hval, N*SHA256_DIGEST_SIZE, cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    printf("SHA256: cudaMemcpy failed.\n");
    err = cudaFree(device_hval);
    if(err != cudaSuccess) {
      printf("SHA256: Failed to free device memory.\n");
    }
    free(host_hval);
    return 1;
  }
  /* should print: 469c24f94970733aa9d3c18ba88b816a5572cdc86286c30107e3ffcb9ef88e05 */
  for(i=0; i<N; i++) {
    offset = i * SHA256_DIGEST_SIZE;
    for(j=0; j<SHA256_DIGEST_SIZE; j++)
      printf("%02X", host_hval[offset+j]);
    printf("\n");
  }

  printf("Timers: SHA1 = %5.3f ms; SHA256 = %5.3f ms\n", sha1_elapsed, sha256_elapsed);

  /* clean up */
  err = cudaFree(device_hval);
  if(err != cudaSuccess) {
    printf("FINAL: Failed to free device memory.\n");
  }
  err = cudaEventDestroy(start);
  if(err != cudaSuccess)
    printf("FINAL: Unable to destroy start event");
  err = cudaEventDestroy(stop);
  if(err != cudaSuccess)
    printf("FINAL: Unable to destroy stop event");
  device_hval = NULL;
  free(host_hval);
  host_hval = NULL;

  return 0;
}
