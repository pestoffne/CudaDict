#include <algorithm>
#include <fstream>

#include <cuda_runtime.h>

#include "common.h"

#include "gpu.h"

const int MAX_THREAD_PER_BLOCK = 1024;  // Для моей видеокарты (GeForce GT 730)
const int THREADS_PER_BLOCK = MAX_THREAD_PER_BLOCK;
const int BLOCK_PER_GRID = 1024;
const int THREADS_PER_GRID = THREADS_PER_BLOCK * BLOCK_PER_GRID;
const int DICT_COUNT = 256;
const int DICT_SIZE = DICT_COUNT * sizeof(unsigned int);

using namespace std;

__global__ void kernel(const char *buff, dict_t dict, unsigned long text_size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = i; j < text_size; j += THREADS_PER_GRID) {
		atomicAdd(&dict[buff[j] + 128], 1);
	}
}

#define RUN(x) (run(x, __FILE__, __LINE__))

__host__ void run(cudaError_t err, const char *file, int line)
{
	if (err) {
#if LANG_RU
		fprintf(stderr, "Ошибка в файле %s на строке %d:\n%s\n%s\n",
#else
		fprintf(stderr, "Error in file %s on line %d:\n%s\n%s\n",
#endif
				file, line, cudaGetErrorName(err), cudaGetErrorString(err));
		EXIT;
	}
}

__host__ dict_t process_gpu(const char *text, unsigned long text_size)
{
	dict_t d_dict = NULL;
	RUN(cudaMalloc((void **)&d_dict, DICT_SIZE));
	RUN(cudaMemset(d_dict, 0, DICT_SIZE));

	char *d_buff = NULL;
	RUN(cudaMalloc((void **)&d_buff, text_size));

	RUN(cudaMemcpy(d_buff, text, text_size, cudaMemcpyHostToDevice));
	kernel <<< BLOCK_PER_GRID, THREADS_PER_BLOCK>>>(d_buff, d_dict, text_size);
	RUN(cudaGetLastError());

	RUN(cudaFree(d_buff));

	dict_t dict = new unsigned int[DICT_COUNT];
	RUN(cudaMemcpy(dict, d_dict, DICT_SIZE, cudaMemcpyDeviceToHost));
	RUN(cudaFree(d_dict));

	return dict;
}
