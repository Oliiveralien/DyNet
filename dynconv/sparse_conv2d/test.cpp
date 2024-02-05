#include <stdio.h>
#include <omp.h>

#ifdef _OPENMP

#define DO_PRAGMA(x) _Pragma ( #x )

#define CPU_1D_KERNEL_LOOP(i, n , thread_count)             \
  DO_PRAGMA(omp parallel for num_threads(thread_count) )        \
  for (int i = 0; i < n; i++)
  
#else 

#define CPU_1D_KERNEL_LOOP(i, n)                            \
  for (int i = 0; i < n; i++)

#endif


int main(int argc, char* argv[]){
    const int n = 12;
    const int num_thread = 6;
    printf("max num thread = %d\n",omp_get_num_procs());
    CPU_1D_KERNEL_LOOP(index,n,num_thread){
        printf("%d pid=%d total_thread = %d\n",index,
                                               omp_get_thread_num(),
                                               omp_get_num_threads());
    }
}