// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "papi.h"

std::vector<StubCall>& stub_papi_calls() {
  static std::vector<StubCall> calls;
  return calls;
}

extern "C" int PAPI_hl_region_begin(const char* region) {
  stub_papi_calls().push_back({"begin", region});
  return PAPI_OK;
}

extern "C" int PAPI_hl_region_end(const char* region) {
  stub_papi_calls().push_back({"end", region});
  return PAPI_OK;
}

extern "C" int PAPI_hl_read(const char* region) {
  stub_papi_calls().push_back({"read", region});
  return PAPI_OK;
}
