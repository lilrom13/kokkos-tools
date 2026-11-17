// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Test stand-in for PAPI's high-level API: records the calls the PAPI
// connector makes instead of reading hardware counters.

#ifndef KOKKOSTOOLS_TEST_STUB_PAPI_H
#define KOKKOSTOOLS_TEST_STUB_PAPI_H

#define PAPI_OK 0

extern "C" {
int PAPI_hl_region_begin(const char* region);
int PAPI_hl_region_end(const char* region);
int PAPI_hl_read(const char* region);
}

#include <string>
#include <vector>

struct StubCall {
  std::string op;    // "begin", "end" or "read"
  std::string name;  // region name passed to PAPI
};

// Calls recorded since the last clear(), in order.
std::vector<StubCall>& stub_papi_calls();

#endif
