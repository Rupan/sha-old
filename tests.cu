#include <stdio.h>

__global__ void kernel( void ) {
}

int main( void ) {
  printf("Running kernel...\n");
  kernel<<<1,1>>>();
  printf( "Exiting...\n" );
  return 0;
}
