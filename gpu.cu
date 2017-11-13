#include <algorithm>
#include <fstream>
#include <unistd.h>

#include <cuda_runtime.h>

#include "gpu.h"

constexpr int MAX_THREAD_PER_BLOCK = 1024;  // Для моей видеокарты (GeForce GT 730)
constexpr int THREADS_PER_BLOCK = MAX_THREAD_PER_BLOCK;
constexpr int BLOCK_PER_GRID = 1024;
constexpr int THREADS_PER_GRID = THREADS_PER_BLOCK * BLOCK_PER_GRID;
constexpr int DICTS_COUNT = THREADS_PER_GRID * 256;
constexpr int DICTS_SIZE = DICTS_COUNT * sizeof(unsigned int);
constexpr int BUFFER_SIZE = 256 * 1024;

#define DIVIDE_CEIL(x, y) (((x) + (y) - 1) / (y))

using namespace std;

__global__ void kernel(const char *buff, dict_t dicts)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = i; j < BUFFER_SIZE; j += THREADS_PER_GRID) {
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

__host__ dict_t process_gpu(const char *path)
{
	ifstream ifs(path, ifstream::binary);

	if (!ifs) {
		fprintf(stderr, "Невозможно прочитать файл.\n");
		exit(EXIT_FAILURE);
	}

	dict_t d_dicts = NULL;
	RUN(cudaMalloc((void **)&d_dicts, DICTS_SIZE));
	RUN(cudaMemset(d_dicts, 0, DICTS_SIZE));

	char *d_buff = NULL;
	RUN(cudaMalloc((void **)&d_buff, BUFFER_SIZE));

	char *h_buff = new char[BUFFER_SIZE];

	do {
		fill(h_buff, h_buff + BUFFER_SIZE, '\0');
		ifs.read(h_buff, BUFFER_SIZE);

		RUN(cudaMemcpy(d_buff, h_buff, BUFFER_SIZE, cudaMemcpyHostToDevice));
		kernel<<<BLOCK_PER_GRID, THREADS_PER_BLOCK>>>(d_buff, d_dicts);
		RUN(cudaGetLastError());
	} while (ifs);

	ifs.close();
	delete h_buff;
	RUN(cudaFree(d_buff));

	sum<<<1, 256>>>(d_dicts);
	RUN(cudaGetLastError());

	dict_t dict = new unsigned int[256];
	RUN(cudaMemcpy(dict, d_dicts, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	RUN(cudaFree(d_dicts));

	dict['\0' + 128] = 0;
	dict['a' + 128] = 1;

	return dict;
}
