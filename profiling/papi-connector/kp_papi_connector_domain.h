// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef _H_KOKKOSP_KERNEL_PAPI_CONNECTOR_INFO
#define _H_KOKKOSP_KERNEL_PAPI_CONNECTOR_INFO

#include <stdio.h>
#include <sys/time.h>
#include <cstring>

#include "papi.h"
#include "utils/kp_event_filter.hpp"

struct SpaceHandle {
  char name[64];
};

/* stack for parallel_for */
std::stack<std::string> parallel_for_name;

/* stack for parallel_reduce */
std::stack<std::string> parallel_reduce_name;

/* stack for parallel_scan */
std::stack<std::string> parallel_scan_name;

/* stack for region names */
std::stack<std::string> region_name;

/* map of profile sections */
std::map<uint32_t, std::string> profile_section;

/* events selected by KOKKOS_TOOLS_PAPI_FILTER (select-all without it) */
KokkosTools::EventFilter::Filter papi_filter;

/* whether each open deep copy is measured */
std::stack<bool> deep_copy_measured;

/* PAPI's high-level API allows PAPIHL_MAX_STACK_SIZE (10, in papi_hl.c)
   nested regions; one more deactivates its output for the whole run. */
constexpr int kMaxPapiDepth = 10;
int papi_depth              = 0;

/* one-time warnings */
bool warned_depth      = false;
bool warned_region_pop = false;
bool warned_for_end    = false;
bool warned_reduce_end = false;
bool warned_scan_end   = false;
bool warned_copy_end   = false;

#endif
