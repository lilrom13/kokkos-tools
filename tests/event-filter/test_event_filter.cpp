// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdlib>
#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "utils/kp_event_filter.hpp"

using namespace KokkosTools::EventFilter;

namespace {
Filter parse(const std::string& text) {
  std::istringstream in(text);
  return Filter::from_stream(in, "f");
}
}  // namespace

TEST(EventFilterParse, CountsRulesPerType) {
  auto f = parse(
      "region: Total\nregion: TimeLoop\nkernel: .*\n"
      "region-path: TimeLoop/AMR/MPI ghosts\ncopy: .*\nsection: IO\n");
  EXPECT_FALSE(f.select_all());
  EXPECT_EQ(f.rule_count(RuleType::Region), 2u);
  EXPECT_EQ(f.rule_count(RuleType::RegionPath), 1u);
  EXPECT_EQ(f.rule_count(RuleType::Kernel), 1u);
  EXPECT_EQ(f.rule_count(RuleType::Copy), 1u);
  EXPECT_EQ(f.rule_count(RuleType::Section), 1u);
}

TEST(EventFilterParse, CommentsBlankLinesAndWhitespace) {
  auto f = parse(
      "# header\n\n   \n  region:   TimeLoop   # trailing comment\n"
      "region: a#b\n");
  EXPECT_EQ(f.rule_count(RuleType::Region), 2u);
}

TEST(EventFilterParse, ColonInsideNameKept) {
  EXPECT_EQ(parse("region: AMR: adapt\n").rule_count(RuleType::Region), 1u);
}

TEST(EventFilterParse, WindowsLineEndings) {
  auto f = parse("region: TimeLoop\r\nkernel: .*\r\n");
  EXPECT_EQ(f.rule_count(RuleType::Region), 1u);
  EXPECT_EQ(f.rule_count(RuleType::Kernel), 1u);
}

TEST(EventFilterParse, EmptyFileSelectsNothing) {
  auto f = parse("# nothing\n");
  EXPECT_FALSE(f.select_all());
  EXPECT_EQ(f.rule_count(RuleType::Region), 0u);
}

TEST(EventFilterParse, DefaultIsSelectAll) {
  EXPECT_TRUE(Filter().select_all());
}

TEST(EventFilterParse, UnknownTypeIsAnErrorWithLine) {
  try {
    parse("region: a\nkernal: b\n");
    FAIL() << "expected FilterError";
  } catch (const FilterError& e) {
    EXPECT_EQ(e.line(), 2);
    EXPECT_THAT(e.what(),
                ::testing::HasSubstr("f: line 2: unknown type 'kernal'"));
    EXPECT_THAT(e.what(),
                ::testing::HasSubstr(
                    "(expected region, region-path, kernel, copy or section)"));
  }
}

TEST(EventFilterParse, MissingColonIsAnError) {
  try {
    parse("TimeLoop\n");
    FAIL() << "expected FilterError";
  } catch (const FilterError& e) {
    EXPECT_EQ(e.line(), 1);
  }
}

TEST(EventFilterParse, EmptyPatternIsAnError) {
  try {
    parse("region:   \n");
    FAIL() << "expected FilterError";
  } catch (const FilterError& e) {
    EXPECT_EQ(e.line(), 1);
  }
}

TEST(EventFilterParse, InvalidRegexIsAnError) {
  try {
    parse("kernel: [unclosed\n");
    FAIL() << "expected FilterError";
  } catch (const FilterError& e) {
    EXPECT_EQ(e.line(), 1);
    EXPECT_THAT(e.what(), ::testing::HasSubstr("invalid pattern '[unclosed'"));
  }
}

TEST(EventFilterFile, MissingFileIsAnError) {
  try {
    Filter::from_file("/nonexistent/filter.txt");
    FAIL() << "expected FilterError";
  } catch (const FilterError& e) {
    EXPECT_EQ(e.line(), 0);
    EXPECT_THAT(e.what(), ::testing::HasSubstr(
                              "/nonexistent/filter.txt: cannot open file"));
  }
}

TEST(EventFilterEnv, UnsetOrEmptyMeansSelectAll) {
  unsetenv("KP_TEST_FILTER");
  EXPECT_TRUE(Filter::from_env("KP_TEST_FILTER").select_all());
  setenv("KP_TEST_FILTER", "", 1);
  EXPECT_TRUE(Filter::from_env("KP_TEST_FILTER").select_all());
  unsetenv("KP_TEST_FILTER");
}
