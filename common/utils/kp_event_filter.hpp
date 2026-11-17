// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Event filter for Kokkos Tools: decides which Kokkos events a tool handles,
// from a file of rules, one per line:
//
//   # comment
//   region: TimeLoop               profile region, matched by name
//   region-path: Total/TimeLoop    profile region, matched by nesting path
//   kernel: .*Flux.*               parallel_for / parallel_reduce /
//   parallel_scan copy: .*                       deep copies (destination View
//   label) section: IO                    profile sections
//
// Patterns are ECMAScript regular expressions matched against the whole
// name. Without a filter, every event is selected.
//
// This header has no dependency on Kokkos or on any measurement library.

#ifndef KOKKOSTOOLS_KP_EVENT_FILTER_HPP
#define KOKKOSTOOLS_KP_EVENT_FILTER_HPP

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace KokkosTools {
namespace EventFilter {

enum class RuleType { Region, RegionPath, Kernel, Copy, Section };

class FilterError : public std::runtime_error {
 public:
  FilterError(const std::string& message, int line)
      : std::runtime_error(message), line_(line) {}
  // Line of the filter file the error refers to, 0 when not line-specific.
  int line() const { return line_; }

 private:
  int line_;
};

namespace Impl {

inline bool is_blank(char c) { return c == ' ' || c == '\t'; }

inline std::string trim(const std::string& s) {
  std::size_t begin = 0;
  std::size_t end   = s.size();
  while (begin < end && is_blank(s[begin])) ++begin;
  while (end > begin && is_blank(s[end - 1])) --end;
  return s.substr(begin, end - begin);
}

// A '#' starts a comment at the beginning of a line or after a blank.
inline std::string strip_comment(const std::string& line) {
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '#' && (i == 0 || is_blank(line[i - 1]))) {
      return line.substr(0, i);
    }
  }
  return line;
}

inline bool parse_type(const std::string& word, RuleType& type) {
  if (word == "region") {
    type = RuleType::Region;
  } else if (word == "region-path") {
    type = RuleType::RegionPath;
  } else if (word == "kernel") {
    type = RuleType::Kernel;
  } else if (word == "copy") {
    type = RuleType::Copy;
  } else if (word == "section") {
    type = RuleType::Section;
  } else {
    return false;
  }
  return true;
}

}  // namespace Impl

struct Rule {
  RuleType type;
  std::string pattern;  // as written in the file
  std::regex regex;
};

class Filter {
 public:
  // Select-all mode: every event is selected.
  Filter() = default;

  static Filter from_stream(std::istream& in, const std::string& source) {
    Filter filter;
    filter.select_all_ = false;
    std::string raw;
    int line_number = 0;
    while (std::getline(in, raw)) {
      ++line_number;
      if (!raw.empty() && raw.back() == '\r') raw.pop_back();
      const std::string line = Impl::trim(Impl::strip_comment(raw));
      if (line.empty()) continue;

      auto fail = [&](const std::string& reason) {
        return FilterError(
            source + ": line " + std::to_string(line_number) + ": " + reason,
            line_number);
      };

      const std::size_t colon = line.find(':');
      if (colon == std::string::npos) {
        throw fail("expected '<type>: <pattern>'");
      }
      const std::string word    = Impl::trim(line.substr(0, colon));
      const std::string pattern = Impl::trim(line.substr(colon + 1));
      RuleType type;
      if (!Impl::parse_type(word, type)) {
        throw fail("unknown type '" + word +
                   "' (expected region, region-path, kernel, copy or section)");
      }
      if (pattern.empty()) {
        throw fail("empty pattern for type '" + word + "'");
      }
      try {
        filter.rules_.push_back(
            Rule{type, pattern, std::regex(pattern, std::regex::ECMAScript)});
      } catch (const std::regex_error& e) {
        throw fail("invalid pattern '" + pattern + "': " + e.what());
      }
    }
    return filter;
  }

  static Filter from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw FilterError(path + ": cannot open file", 0);
    return from_stream(in, path);
  }

  // Unset or empty variable: select-all mode.
  static Filter from_env(const char* variable) {
    const char* path = std::getenv(variable);
    if (path == nullptr || path[0] == '\0') return Filter();
    return from_file(path);
  }

  bool select_all() const { return select_all_; }

  std::size_t rule_count(RuleType type) const {
    std::size_t n = 0;
    for (const auto& rule : rules_) {
      if (rule.type == type) ++n;
    }
    return n;
  }

 private:
  bool select_all_ = true;
  std::vector<Rule> rules_;
};

}  // namespace EventFilter
}  // namespace KokkosTools

#endif  // KOKKOSTOOLS_KP_EVENT_FILTER_HPP
