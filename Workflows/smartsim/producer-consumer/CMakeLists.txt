cmake_minimum_required(VERSION 3.12)
project(WorkflowExample)

find_package(MPI REQUIRED)

set(SMARTREDIS_INSTALL_DIR "" CACHE STRING "SmartRedis install directory")
find_library(SR_LIB smartredis PATHS ${SMARTREDIS_INSTALL_DIR}/lib64 NO_DEFAULT_PATH REQUIRED)

add_executable(sim sim.cpp)
target_link_libraries(sim PUBLIC ${SR_LIB} MPI::MPI_CXX)
target_include_directories(sim PUBLIC ${SMARTREDIS_INSTALL_DIR}/include)

install(TARGETS sim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
