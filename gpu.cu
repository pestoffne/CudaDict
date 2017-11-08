#include <fstream>

#include <cuda_runtime.h>

#include "common.h"

constexpr int MAX_THREAD_PER_BLOCK = 1024;  // Для моей видеокарты (GeForce GT 730)
constexpr int THREADS_PER_BLOCK = MAX_THREAD_PER_BLOCK;  // Равно значению blockDim.x
constexpr int BLOCK_PER_GRID = 1024;
constexpr int THREADS_PER_GRID = THREADS_PER_BLOCK * BLOCK_PER_GRID;
constexpr int DICTS_COUNT = THREADS_PER_GRID * 256;
constexpr int DICTS_SIZE = DICTS_COUNT * sizeof(unsigned int);
constexpr int BUFFER_SIZE = 256 * 1024;

#define DIVIDE_CEIL(x, y) (((x) + (y) - 1) / (y))

__global__ void kernel(const char *buff, dict_t dicts)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = i; j < BUFFER_SIZE; j += THREADS_PER_GRID) {
		dicts[buff[j] + 128 + 256 * i]++;
	}
}

__host__ dict_t process_gpu(const char *path)
{
	ifstream ifs(path, ifstream::binary);

	if (!ifs) {
		fprintf(stderr, "Невозможно прочитать файл.\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t err = cudaSuccess;

	dict_t d_dicts = NULL;
	err = cudaMalloc((void **)&d_dicts, DICTS_SIZE);

	if (err) {
		fprintf(stderr,
				"Ошибка при выделении памяти для словаря на видеокарте: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemset(d_dicts, 0, DICTS_SIZE);

	if (err) {
		fprintf(stderr, "Ошибка при обнулении словаря на видеокарте: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	char *d_buff = NULL;
	err = cudaMalloc((void **)&d_buff, BUFFER_SIZE);

	if (err) {
		fprintf(stderr,
				"Ошибка при выделении памяти для буфера на видеокарте: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	char *h_buff = new char[BUFFER_SIZE];

	for (int i = BUFFER_SIZE; i >= 0; --i) {
		h_buff[i] = '\0';
	}

	for (;;) {
		ifs.read(h_buff, BUFFER_SIZE);

		err = cudaMemcpy(d_buff, h_buff, BUFFER_SIZE, cudaMemcpyHostToDevice);

		if (err) {
			fprintf(stderr, "Ошибка при копировании буфера из оперативной памяти в"
					" память видеокарты: %s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		kernel<<<BLOCK_PER_GRID, THREADS_PER_BLOCK>>>(d_buff, d_dicts);
		err = cudaGetLastError();

		if (err) {
			fprintf(stderr, "Ошибка при запуске вычислений на видеокарте: %s\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		if (!ifs) {
			break;
		}

		for (int i = BUFFER_SIZE; i >= 0; --i) {
			h_buff[i] = '\0';
		}
	}

	ifs.close();
	delete h_buff;

	err = cudaFree(d_buff);

	if (err) {
		fprintf(stderr, "Ошибка при освобождении буфера: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dict_t h_dicts = new unsigned int[DICTS_SIZE];
	err = cudaMemcpy(h_dicts, d_dicts, DICTS_SIZE, cudaMemcpyDeviceToHost);

	if (err) {
		fprintf(stderr,
				"Ошибка при копировании словаря из памяти видеокарты в оперативную память: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_dicts);

	if (err) {
		fprintf(stderr, "Ошибка при освобождении словаря: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dict_t dict = new unsigned int[256];

	for (int i = 256; i >= 0; --i) {
		dict[i] = h_dicts[i];

		for (int j = THREADS_PER_GRID; j > 0; --j) {
			dict[i] += h_dicts[i + 256 * j];
		}
	}

	delete h_dicts;

	return dict;
}

__host__ int main(int argc, char **argv)
{
	run(process_gpu, argc, argv);
	return 0;
}
