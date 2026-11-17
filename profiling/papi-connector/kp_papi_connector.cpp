// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <stdio.h>
#include <inttypes.h>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <string>
#include <iostream>
#include <stack>
#include <sstream>
#include <map>

#include "impl/Kokkos_Profiling_C_Interface.h"
#include "kp_papi_connector_domain.h"

namespace {

using KokkosTools::EventFilter::EventType;
using KokkosTools::EventFilter::Filter;
using KokkosTools::EventFilter::FilterError;

const char* kFilterVariable = "KOKKOS_TOOLS_PAPI_FILTER";

void warn_once(bool& warned, const char* message) {
  if (!warned) {
    warned = true;
    printf("%s\n", message);
  }
}

// Opens a PAPI region, unless that would exceed PAPI's nesting limit.
bool papi_begin(const std::string& name) {
  if (papi_depth >= kMaxPapiDepth) {
    warn_once(warned_depth,
              "KokkosP: PAPI connector: more than 10 nested measured events, "
              "deeper ones are not measured (PAPI limit)");
    return false;
  }
  ++papi_depth;
  PAPI_hl_region_begin(name.c_str());
  return true;
}

void papi_end(const std::string& name) {
  if (papi_depth > 0) --papi_depth;
  PAPI_hl_region_end(name.c_str());
}

// Pushes the PAPI name of a measured event, or "" for an event that is not
// measured, so that the matching end always pops its own entry.
void begin_kernel(std::stack<std::string>& stack, const char* kind,
                  const char* name) {
  const std::string papi_name = std::string(kind) + ":" + name;
  if (papi_filter.selected(EventType::Kernel, name) && papi_begin(papi_name)) {
    stack.push(papi_name);
  } else {
    stack.push("");
  }
}

void end_kernel(std::stack<std::string>& stack, bool& warned,
                const char* message) {
  if (stack.empty()) {
    warn_once(warned, message);
    return;
  }
  const std::string papi_name = stack.top();
  stack.pop();
  if (!papi_name.empty()) papi_end(papi_name);
}

void reset_state() {
  papi_filter          = Filter();
  parallel_for_name    = {};
  parallel_reduce_name = {};
  parallel_scan_name   = {};
  region_name          = {};
  deep_copy_measured   = {};
  profile_section.clear();
  global_fences     = true;
  papi_depth        = 0;
  warned_depth      = false;
  warned_region_pop = false;
  warned_for_end    = false;
  warned_reduce_end = false;
  warned_scan_end   = false;
  warned_copy_end   = false;
}

}  // namespace

extern "C" void kokkosp_init_library(const int loadSeq,
                                     const uint64_t interfaceVer,
                                     const uint32_t /*devInfoCount*/,
                                     void* /*deviceInfo*/) {
  printf("-----------------------------------------------------------\n");
  printf("KokkosP: PAPI Connector (sequence is %d, version: %llu)\n", loadSeq,
         static_cast<unsigned long long>(interfaceVer));
  printf("-----------------------------------------------------------\n");

  try {
    papi_filter = Filter::from_env(kFilterVariable);
  } catch (const FilterError& e) {
    fprintf(stderr, "KokkosP: PAPI connector: error in %s: %s\n",
            kFilterVariable, e.what());
    exit(EXIT_FAILURE);
  }
  const char* fences = getenv("KOKKOS_TOOLS_GLOBALFENCES");
  global_fences      = fences == nullptr || atoi(fences) != 0;

  if (!papi_filter.select_all()) {
    using KokkosTools::EventFilter::RuleType;
    printf(
        "KokkosP: PAPI connector: filter %s: %zu region, %zu region-path, "
        "%zu kernel, %zu copy, %zu section rules\n",
        getenv(kFilterVariable), papi_filter.rule_count(RuleType::Region),
        papi_filter.rule_count(RuleType::RegionPath),
        papi_filter.rule_count(RuleType::Kernel),
        papi_filter.rule_count(RuleType::Copy),
        papi_filter.rule_count(RuleType::Section));
  }

  /* The following advanced functions of PAPI's high-level API are not part
   * of the official release. But they might be introduced in later PAPI
   * releases. PAPI_hl_init is now called from the first PAPI_hl_region_begin
   * call.
   */
  // PAPI_hl_init();
  /* set default values */
  // PAPI_hl_set_events("perf::TASK-CLOCK,PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_FP_OPS");
}

extern "C" void kokkosp_request_tool_settings(
    const uint32_t, Kokkos_Tools_ToolSettings* settings) {
  settings->requires_global_fencing = global_fences;
}

extern "C" void kokkosp_finalize_library() {
  /* The following advanced functions of PAPI's high-level API are not part
   * of the official release. But they might be introduced in later PAPI
   * releases. PAPI_hl_print_output is registered by the "atexit" function and
   * will be called at process termination, see
   * http://man7.org/linux/man-pages/man3/atexit.3.html.
   */
  // PAPI_hl_print_output();
  // PAPI_hl_finalize();

  printf("-----------------------------------------------------------\n");
  printf("KokkosP: Finalization of PAPI Connector. Complete.\n");
  printf("-----------------------------------------------------------\n");

  reset_state();
}

extern "C" void kokkosp_begin_parallel_for(const char* name, uint32_t /*devid*/,
                                           uint64_t* /*kernid*/) {
  begin_kernel(parallel_for_name, "kokkosp_parallel_for", name);
}
extern "C" void kokkosp_end_parallel_for(uint64_t /*kernid*/) {
  end_kernel(parallel_for_name, warned_for_end,
             "Begin callback of parallel_for does not exist!");
}

extern "C" void kokkosp_begin_parallel_reduce(const char* name,
                                              uint32_t /*devid*/,
                                              uint64_t* /*kernid*/) {
  begin_kernel(parallel_reduce_name, "kokkosp_parallel_reduce", name);
}
extern "C" void kokkosp_end_parallel_reduce(uint64_t /*kernid*/) {
  end_kernel(parallel_reduce_name, warned_reduce_end,
             "Begin callback of parallel_reduce does not exist!");
}

extern "C" void kokkosp_begin_parallel_scan(const char* name,
                                            uint32_t /*devid*/,
                                            uint64_t* /*kernid*/) {
  begin_kernel(parallel_scan_name, "kokkosp_parallel_scan", name);
}
extern "C" void kokkosp_end_parallel_scan(uint64_t /*kernid*/) {
  end_kernel(parallel_scan_name, warned_scan_end,
             "Begin callback of parallel_scan does not exist!");
}

extern "C" void kokkosp_push_profile_region(const char* name) {
  const std::string papi_name = std::string("kokkosp_profile_region:") + name;
  if (papi_filter.push_region(name) && papi_begin(papi_name)) {
    region_name.push(papi_name);
  } else {
    region_name.push("");
  }
}

extern "C" void kokkosp_profile_event(const char* /*name*/) {
  if (!region_name.empty() && !region_name.top().empty()) {
    PAPI_hl_read(region_name.top().c_str());
  }
}

extern "C" void kokkosp_pop_profile_region() {
  if (region_name.empty()) {
    warn_once(warned_region_pop,
              "KokkosP: PAPI connector: pop without matching push");
    return;
  }
  const std::string papi_name = region_name.top();
  region_name.pop();
  papi_filter.pop_region();
  if (!papi_name.empty()) papi_end(papi_name);
}

extern "C" void kokkosp_begin_deep_copy(
    SpaceHandle /*dst_handle*/, const char* dst_name, const void* /*dst_ptr*/,
    SpaceHandle /*src_handle*/, const char* /*src_name*/,
    const void* /*src_ptr*/, uint64_t /*size*/) {
  const bool measured =
      papi_filter.selected(EventType::Copy, dst_name ? dst_name : "") &&
      papi_begin("kokkosp_deep_copy");
  deep_copy_measured.push(measured);
}
extern "C" void kokkosp_end_deep_copy() {
  if (deep_copy_measured.empty()) {
    warn_once(warned_copy_end,
              "KokkosP: PAPI connector: deep copy end without begin");
    return;
  }
  const bool measured = deep_copy_measured.top();
  deep_copy_measured.pop();
  if (measured) papi_end("kokkosp_deep_copy");
}

extern "C" void kokkosp_create_profile_section(const char* name,
                                               uint32_t* sec_id) {
  // printf("kokkosp_create_profile_section: %s %d\n", name, *sec_id);
  profile_section.insert(std::pair<uint32_t, std::string>(*sec_id, name));
}
extern "C" void kokkosp_destroy_profile_section(uint32_t sec_id) {
  // printf("kokkosp_destroy_profile_section: %d\n", sec_id);
  std::map<uint32_t, std::string>::iterator it;
  it = profile_section.find(sec_id);
  if (it != profile_section.end()) profile_section.erase(it);
}

extern "C" void kokkosp_start_profile_section(uint32_t sec_id) {
  // printf("kokkosp_start_profile_section: %d\n", sec_id);
  std::map<uint32_t, std::string>::iterator it;
  it = profile_section.find(sec_id);
  if (it != profile_section.end()) {
    if (!papi_filter.selected(EventType::Section, it->second)) return;
    std::stringstream ss;
    ss << it->second << ":" << sec_id;
    region_name.push(ss.str());
    papi_begin(ss.str());
  }
}
extern "C" void kokkosp_stop_profile_section(uint32_t sec_id) {
  // printf("kokkosp_stop_profile_section: %d\n", sec_id);
  std::map<uint32_t, std::string>::iterator it;
  it = profile_section.find(sec_id);
  if (it != profile_section.end()) {
    if (!papi_filter.selected(EventType::Section, it->second)) return;
    std::stringstream ss;
    ss << it->second << ":" << sec_id;
    region_name.push(ss.str());
    papi_end(ss.str());
  }
}
