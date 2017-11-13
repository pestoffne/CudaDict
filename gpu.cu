#include <algorithm>
#include <fstream>
#include <unistd.h>

#include <cuda_runtime.h>

#include "gpu.h"

constexpr int MAX_THREAD_PER_BLOCK = 1024;  // Для моей видеокарты (GeForce GT 730)
constexpr int THREADS_PER_BLOCK = MAX_THREAD_PER_BLOCK;
constexpr int BLOCK_PER_GRID = 256;
constexpr int THREADS_PER_GRID = THREADS_PER_BLOCK * BLOCK_PER_GRID;
constexpr int DICTS_COUNT = THREADS_PER_GRID * 256;
constexpr int DICTS_SIZE = DICTS_COUNT * sizeof(unsigned int);

#define DIVIDE_CEIL(x, y) (((x) + (y) - 1) / (y))

using namespace std;

__global__ void kernel(const char *buff, dict_t dicts, unsigned long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = i; j < size; j += THREADS_PER_GRID) {
		dicts[buff[j] + 128 + 256 * i]++;
	}
}

__global__ void sum(dict_t dicts)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = 1; j < THREADS_PER_GRID; j++) {
		dicts[i] += dicts[i + 256 * j];
	}
}

#define RUN(x) (run(x, __FILE__, __LINE__))

__host__ void run(cudaError_t err, const char *file, int line)
{
	if (err) {
		fprintf(stderr, "Ошибка в файле %s на строке %d:\n%s\n%s\n",
				file, line, cudaGetErrorName(err), cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__host__ dict_t process_gpu(const char *text, unsigned long text_size)
{
	dict_t d_dicts = NULL;
	RUN(cudaMalloc((void **)&d_dicts, DICTS_SIZE));
	RUN(cudaMemset(d_dicts, 0, DICTS_SIZE));

	char *d_buff = NULL;
	RUN(cudaMalloc((void **)&d_buff, text_size));

	RUN(cudaMemcpy(d_buff, text, text_size, cudaMemcpyHostToDevice));
	kernel<<<BLOCK_PER_GRID, THREADS_PER_BLOCK>>>(d_buff, d_dicts, text_size);
	RUN(cudaGetLastError());

	RUN(cudaFree(d_buff));

	sum<<<1, 256>>>(d_dicts);
	RUN(cudaGetLastError());

	dict_t dict = new unsigned int[256];
	RUN(cudaMemcpy(dict, d_dicts, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	RUN(cudaFree(d_dicts));

	return dict;
}
