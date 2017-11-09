#pragma once

using namespace std;

typedef unsigned int *dict_t;

const char *parse(int argc, char **argv);
void print_result(const char *path, dict_t dict);
