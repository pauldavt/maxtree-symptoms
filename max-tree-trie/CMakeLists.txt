cmake_minimum_required(VERSION 3.18)

project(mta)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)


set(CMAKE_CXX_FLAGS  "-Wfatal-errors -march=native -O3 -pedantic -std=c++14 ${CMAKE_CXX_FLAGS}")

add_executable(trie_queue tests/trie_queue.cc)
add_executable(radix_sort tests/radix_sort.cc)
add_executable(maxtree_trie tests/maxtree_trie.cc src/bit_array.cc)
add_executable(area_example area_example.cc src/bit_array.cc)

add_library(mta SHARED
  src/bit_array.cc
  module.cc)

target_include_directories(mta PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(mta pthread atomic)

set_target_properties(mta PROPERTIES PREFIX "")
set_target_properties(mta PROPERTIES OUTPUT_NAME "mta")
set_target_properties(mta PROPERTIES SUFFIX ".so")



