cmake_minimum_required(VERSION 3.12)
project(WorkflowExample CXX)

# Need C++17 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find MPI
find_package(MPI REQUIRED)

# Find Dragon 
if(NOT DEFINED DRAGON_ROOT)
  message(FATAL_ERROR
    "DRAGON_ROOT is not set. Run ./config.sh, or invoke cmake with "
    "-DDRAGON_ROOT=/path/to/site-packages/dragon")
endif()

set(DRAGON_INCLUDE_DIR "${DRAGON_ROOT}/include")
set(DRAGON_LIBRARY_DIR "${DRAGON_ROOT}/lib")

find_library(DRAGON_LIBRARY
  NAMES dragon
  HINTS "${DRAGON_LIBRARY_DIR}"
  NO_DEFAULT_PATH
  REQUIRED)

message(STATUS "Dragon root    : ${DRAGON_ROOT}")
message(STATUS "Dragon include : ${DRAGON_INCLUDE_DIR}")
message(STATUS "Dragon library : ${DRAGON_LIBRARY}")

# Build
add_executable(sim sim.cpp)
target_compile_definitions(sim PRIVATE _GLIBCXX_USE_CXX11_ABI=0) # Needed to resolve conflict with dragon library shipped with wheel
target_include_directories(sim PRIVATE ${DRAGON_INCLUDE_DIR})
target_link_libraries(sim PRIVATE
  ${DRAGON_LIBRARY}
  MPI::MPI_CXX
  ${CMAKE_DL_LIBS})
install(TARGETS sim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
