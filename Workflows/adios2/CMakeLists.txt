cmake_minimum_required(VERSION 3.12)
project(WorkflowExample)

if(NOT TARGET adios2_core)
  set(_components CXX)

  find_package(MPI COMPONENTS C)
  if(MPI_FOUND)
    # Workaround for various MPI implementations forcing the link of C++ bindings
    add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

    list(APPEND _components MPI)
  endif()

  find_package(ADIOS2 REQUIRED COMPONENTS ${_components})
endif()

add_executable(sim sim.cpp)
target_link_libraries(sim adios2::cxx_mpi MPI::MPI_CXX)
install(TARGETS sim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
