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

TEST(EventFilterSelect, WholeNameMatch) {
  auto f = parse("kernel: Hydro.*\n");
  EXPECT_TRUE(f.selected(EventType::Kernel, "HydroFlux"));
  EXPECT_FALSE(f.selected(EventType::Kernel, "MyHydroFlux"));
}

TEST(EventFilterSelect, TypeWithoutRulesIsNeverSelected) {
  auto f = parse("region: TimeLoop\n");
  EXPECT_FALSE(f.selected(EventType::Kernel, "anything"));
  EXPECT_FALSE(f.selected(EventType::Copy, "view_a"));
  EXPECT_FALSE(f.selected(EventType::Section, "IO"));
}

TEST(EventFilterSelect, SelectAllSelectsEverything) {
  Filter f;
  EXPECT_TRUE(f.selected(EventType::Kernel, "k"));
  EXPECT_TRUE(f.push_region("r"));
}

TEST(EventFilterSelect, EmptyFileSelectsNothing) {
  auto f = parse("\n");
  EXPECT_FALSE(f.selected(EventType::Kernel, "k"));
  EXPECT_FALSE(f.push_region("r"));
}

TEST(EventFilterSelect, RegexMetacharactersInLabels) {
  const std::string label = "Kokkos::View::initialization [rho] via memset";
  auto literal            = parse("kernel: " + label + "\n");
  EXPECT_FALSE(literal.selected(EventType::Kernel, label));
  auto escaped =
      parse("kernel: Kokkos::View::initialization \\[rho\\] via memset\n");
  EXPECT_TRUE(escaped.selected(EventType::Kernel, label));
}

TEST(EventFilterSelect, CommentAndColonRulesMatch) {
  auto f =
      parse("region: AMR: adapt\nregion: a#b\nregion: TimeLoop   # comment\n");
  EXPECT_TRUE(f.push_region("AMR: adapt"));
  f.pop_region();
  EXPECT_TRUE(f.push_region("a#b"));
  f.pop_region();
  EXPECT_TRUE(f.push_region("TimeLoop"));
  f.pop_region();
}

TEST(EventFilterRegions, NameAndPathRules) {
  auto f = parse("region: Total\nregion-path: Total/TimeLoop/AMR/MPI ghosts\n");
  EXPECT_TRUE(f.push_region("Total"));
  EXPECT_FALSE(f.push_region("TimeLoop"));
  EXPECT_FALSE(f.push_region("MPI ghosts"));  // Total/TimeLoop/MPI ghosts
  f.pop_region();
  EXPECT_FALSE(f.push_region("AMR"));
  EXPECT_TRUE(f.push_region("MPI ghosts"));  // Total/TimeLoop/AMR/MPI ghosts
  f.pop_region();
  f.pop_region();
  f.pop_region();
  f.pop_region();
  EXPECT_EQ(f.open_regions(), 0u);
}

TEST(EventFilterRegions, SameNameNestedInItself) {
  auto f = parse("region-path: TimeLoop/TimeLoop\n");
  EXPECT_FALSE(f.push_region("TimeLoop"));
  EXPECT_TRUE(f.push_region("TimeLoop"));
  f.pop_region();
  EXPECT_FALSE(f.push_region("Other"));
  f.pop_region();
  f.pop_region();
  EXPECT_EQ(f.open_regions(), 0u);
}

TEST(EventFilterRegions, PopOnEmptyStackIsHarmless) {
  auto f = parse("region: a\n");
  f.pop_region();
  EXPECT_EQ(f.open_regions(), 0u);
  EXPECT_TRUE(f.push_region("a"));
}

TEST(EventFilterCache, RepeatedQueriesGiveTheSameAnswer) {
  auto f = parse("kernel: k[0-9]+\n");
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(f.selected(EventType::Kernel, "k42"));
    EXPECT_FALSE(f.selected(EventType::Kernel, "x"));
  }
}

TEST(EventFilterCache, AnswersStayCorrectPastTheCap) {
  auto f = parse("kernel: step_[0-9]*7\n");
  for (std::size_t i = 0; i < Filter::cache_cap + 10; ++i) {
    ASSERT_EQ(f.selected(EventType::Kernel, "step_" + std::to_string(i)),
              i % 10 == 7)
        << "i = " << i;
  }
}
