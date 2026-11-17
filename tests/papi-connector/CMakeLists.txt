# The PAPI connector is compiled against a stub PAPI that records calls, so
# these tests need neither PAPI nor hardware counters.
kp_add_executable_and_test(
    TARGET_NAME test_papi_connector_filter
    SOURCE_FILE test_papi_connector_filter.cpp
)
target_sources(
    test_papi_connector_filter
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stub/stub_papi.cpp
        ${PROJECT_SOURCE_DIR}/profiling/papi-connector/kp_papi_connector.cpp
)
target_include_directories(
    test_papi_connector_filter
    BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stub
)
