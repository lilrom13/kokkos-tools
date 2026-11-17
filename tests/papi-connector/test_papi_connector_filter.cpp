// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Behaviour of the PAPI connector with and without KOKKOS_TOOLS_PAPI_FILTER.
// The connector's callbacks are called directly; a stub PAPI records which
// PAPI regions they open and close.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "papi.h"  // the stub

// Same layout as the connector's SpaceHandle.
struct SpaceHandle {
  char name[64];
};

extern "C" {
void kokkosp_init_library(const int, const uint64_t, const uint32_t, void*);
void kokkosp_finalize_library();
void kokkosp_begin_parallel_for(const char*, uint32_t, uint64_t*);
void kokkosp_end_parallel_for(uint64_t);
void kokkosp_begin_parallel_reduce(const char*, uint32_t, uint64_t*);
void kokkosp_end_parallel_reduce(uint64_t);
void kokkosp_begin_parallel_scan(const char*, uint32_t, uint64_t*);
void kokkosp_end_parallel_scan(uint64_t);
void kokkosp_push_profile_region(const char*);
void kokkosp_pop_profile_region();
void kokkosp_profile_event(const char*);
void kokkosp_begin_deep_copy(SpaceHandle, const char*, const void*, SpaceHandle,
                             const char*, const void*, uint64_t);
void kokkosp_end_deep_copy();
void kokkosp_create_profile_section(const char*, uint32_t*);
void kokkosp_destroy_profile_section(uint32_t);
void kokkosp_start_profile_section(uint32_t);
void kokkosp_stop_profile_section(uint32_t);
}

namespace {

const char* kFilterVariable = "KOKKOS_TOOLS_PAPI_FILTER";

// Writes `text` to a filter file and points the variable at it; nullptr
// unsets the variable. Then initializes the connector with a clean log.
void init(const char* text) {
  if (text == nullptr) {
    unsetenv(kFilterVariable);
  } else {
    const std::string path =
        ::testing::TempDir() + "papi_connector_filter_test.txt";
    std::ofstream(path) << text;
    setenv(kFilterVariable, path.c_str(), 1);
  }
  stub_papi_calls().clear();
  kokkosp_init_library(0, 0, 0, nullptr);
}

void fini() {
  kokkosp_finalize_library();
  unsetenv(kFilterVariable);
}

std::vector<std::string> calls() {
  std::vector<std::string> out;
  for (const auto& c : stub_papi_calls()) out.push_back(c.op + ":" + c.name);
  return out;
}

int count_occurrences(const std::string& text, const std::string& what) {
  int n = 0;
  for (std::size_t pos = text.find(what); pos != std::string::npos;
       pos             = text.find(what, pos + what.size()))
    ++n;
  return n;
}

SpaceHandle host() { return SpaceHandle{"Host"}; }

}  // namespace

TEST(PapiConnectorFilter, NoFilterMeasuresEverythingAsToday) {
  init(nullptr);
  uint64_t k = 0;
  kokkosp_push_profile_region("TimeLoop");
  kokkosp_begin_parallel_for("flux", 0, &k);
  kokkosp_end_parallel_for(k);
  kokkosp_begin_deep_copy(host(), "dst", nullptr, host(), "src", nullptr, 8);
  kokkosp_end_deep_copy();
  kokkosp_profile_event("mark");
  kokkosp_pop_profile_region();
  fini();
  EXPECT_EQ(calls(),
            (std::vector<std::string>{
                "begin:kokkosp_profile_region:TimeLoop",
                "begin:kokkosp_parallel_for:flux",
                "end:kokkosp_parallel_for:flux", "begin:kokkosp_deep_copy",
                "end:kokkosp_deep_copy", "read:kokkosp_profile_region:TimeLoop",
                "end:kokkosp_profile_region:TimeLoop"}));
}

TEST(PapiConnectorFilter, OnlySelectedRegions) {
  init("region: Total\nregion: TimeLoop\n");
  uint64_t k = 0;
  kokkosp_push_profile_region("Total");
  kokkosp_push_profile_region("TimeLoop");
  kokkosp_push_profile_region("AMR");
  kokkosp_begin_parallel_for("flux", 0, &k);
  kokkosp_end_parallel_for(k);
  kokkosp_profile_event("mark");  // innermost region AMR not measured: no read
  kokkosp_pop_profile_region();   // AMR
  kokkosp_pop_profile_region();   // TimeLoop
  kokkosp_pop_profile_region();   // Total
  fini();
  EXPECT_EQ(calls(),
            (std::vector<std::string>{"begin:kokkosp_profile_region:Total",
                                      "begin:kokkosp_profile_region:TimeLoop",
                                      "end:kokkosp_profile_region:TimeLoop",
                                      "end:kokkosp_profile_region:Total"}));
}

TEST(PapiConnectorFilter, SelectedKernelInsideUnselectedRegion) {
  init("kernel: flux\n");
  uint64_t k1 = 0, k2 = 0;
  kokkosp_push_profile_region("TimeLoop");
  kokkosp_begin_parallel_reduce("dt", 0, &k1);
  kokkosp_end_parallel_reduce(k1);
  kokkosp_begin_parallel_for("flux", 0, &k2);
  kokkosp_end_parallel_for(k2);
  kokkosp_pop_profile_region();
  fini();
  EXPECT_EQ(calls(),
            (std::vector<std::string>{"begin:kokkosp_parallel_for:flux",
                                      "end:kokkosp_parallel_for:flux"}));
}

TEST(PapiConnectorFilter, NestedSkippedKernelsKeepStacksBalanced) {
  init("kernel: outer\n");
  uint64_t a = 0, b = 0;
  kokkosp_begin_parallel_for("outer", 0, &a);
  kokkosp_begin_parallel_for("inner", 0, &b);
  kokkosp_end_parallel_for(b);
  kokkosp_end_parallel_for(a);
  fini();
  EXPECT_EQ(calls(),
            (std::vector<std::string>{"begin:kokkosp_parallel_for:outer",
                                      "end:kokkosp_parallel_for:outer"}));
}

TEST(PapiConnectorFilter, ScanKernelsFiltered) {
  init("kernel: prefix\n");
  uint64_t a = 0, b = 0;
  kokkosp_begin_parallel_scan("prefix", 0, &a);
  kokkosp_end_parallel_scan(a);
  kokkosp_begin_parallel_scan("other", 0, &b);
  kokkosp_end_parallel_scan(b);
  fini();
  EXPECT_EQ(calls(),
            (std::vector<std::string>{"begin:kokkosp_parallel_scan:prefix",
                                      "end:kokkosp_parallel_scan:prefix"}));
}

TEST(PapiConnectorFilter, CopiesSelectedByDestinationLabel) {
  init("copy: view_b\n");
  kokkosp_begin_deep_copy(host(), "view_a", nullptr, host(), "x", nullptr, 8);
  kokkosp_end_deep_copy();
  kokkosp_begin_deep_copy(host(), "view_b", nullptr, host(), "x", nullptr, 8);
  kokkosp_end_deep_copy();
  fini();
  EXPECT_EQ(calls(), (std::vector<std::string>{"begin:kokkosp_deep_copy",
                                               "end:kokkosp_deep_copy"}));
}

TEST(PapiConnectorFilter, UnbalancedPopWarnsOnceAndCallsNothing) {
  init("region: a\n");
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  kokkosp_pop_profile_region();
  kokkosp_pop_profile_region();
  std::string out = testing::internal::GetCapturedStdout();
  out += testing::internal::GetCapturedStderr();
  fini();
  EXPECT_TRUE(calls().empty());
  EXPECT_EQ(count_occurrences(
                out, "KokkosP: PAPI connector: pop without matching push"),
            1);
}

TEST(PapiConnectorFilter, DepthGuardStopsAtTenLevels) {
  init(nullptr);
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  for (int i = 0; i < 12; ++i)
    kokkosp_push_profile_region(("L" + std::to_string(i)).c_str());
  for (int i = 0; i < 12; ++i) kokkosp_pop_profile_region();
  std::string out = testing::internal::GetCapturedStdout();
  out += testing::internal::GetCapturedStderr();
  fini();
  auto c = calls();
  ASSERT_EQ(c.size(), 20u);  // L0..L9 begin and end; L10 and L11 skipped
  EXPECT_EQ(c[9], "begin:kokkosp_profile_region:L9");
  EXPECT_EQ(c[10], "end:kokkosp_profile_region:L9");
  EXPECT_EQ(count_occurrences(out, "more than 10 nested measured events"), 1);
}

TEST(PapiConnectorFilter, FinalizeResetsState) {
  init("region: a\n");
  kokkosp_push_profile_region("b");  // left open on purpose
  fini();
  init(nullptr);
  kokkosp_pop_profile_region();  // must not close "b" from the last session
  kokkosp_push_profile_region("c");
  kokkosp_pop_profile_region();
  fini();
  EXPECT_EQ(calls(),
            (std::vector<std::string>{"begin:kokkosp_profile_region:c",
                                      "end:kokkosp_profile_region:c"}));
}

TEST(PapiConnectorFilterDeathTest, BadFilterFileExitsWithLine) {
  EXPECT_EXIT(init("region: a\nkernal: b\n"),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              "line 2: unknown type 'kernal'");
}

TEST(PapiConnectorFilterDeathTest, MissingFilterFileExits) {
  EXPECT_EXIT(
      {
        setenv(kFilterVariable, "/nonexistent/filter.txt", 1);
        kokkosp_init_library(0, 0, 0, nullptr);
      },
      ::testing::ExitedWithCode(EXIT_FAILURE), "cannot open file");
}
