%%writefile lab/usm_lab.cpp
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>

using namespace sycl;

static const int N = 1024;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  // Initialize 2 arrays on host
  int *data1 = static_cast<int *>(malloc_host(N * sizeof(int), q));
  int *data2 = static_cast<int *>(malloc_host(N * sizeof(int), q));
  for (int i = 0; i < N; i++) {
    data1[i] = 25;
    data2[i] = 49;
  }
    
  // STEP 1: Create USM device allocation for data1 and data2
  int *data1_dev = static_cast<int *>(malloc_device(N * sizeof(int), q));
  int *data2_dev = static_cast<int *>(malloc_device(N * sizeof(int), q));

  // STEP 2: Copy data1 and data2 to USM device allocation
  q.memcpy(data1_dev, data1, N * sizeof(int)).wait();
  q.memcpy(data2_dev, data2, N * sizeof(int)).wait();

  // STEP 3: Write kernel code to update data1 on device with sqrt of value
  q.parallel_for(N, [=](auto i) {
    data1_dev[i] = static_cast<int>(sqrt(static_cast<float>(data1_dev[i])));
  }).wait();

  // STEP 3: Write kernel code to update data2 on device with sqrt of value
  q.parallel_for(N, [=](auto i) {
    data2_dev[i] = static_cast<int>(sqrt(static_cast<float>(data2_dev[i])));
  }).wait();

  // STEP 5: Write kernel code to add data2 on device to data1
  q.parallel_for(N, [=](auto i) {
    data1_dev[i] += data2_dev[i];
  }).wait();

  // STEP 6: Copy data1 on device to host
  q.memcpy(data1, data1_dev, N * sizeof(int)).wait();

  // Verify results
  int fail = 0;
  for (int i = 0; i < N; i++) {
    if (data1[i] != 12) {
      fail = 1;
      break;
    }
  }
  if (fail == 1) std::cout << " FAIL"; else std::cout << " PASS";
  std::cout << "\n";

  // STEP 7: Free USM device allocations
  free(data1_dev, q);
  free(data2_dev, q);
  free(data1, q);
  free(data2, q);

  return 0;
}
