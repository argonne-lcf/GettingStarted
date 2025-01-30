#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <cassert>

#include <mpi.h>

#include <CL/opencl.h>

#define MAX_PLATFORM_SIZE 256
#define MAX_DEVICE_SIZE 256

// copied from intel_ocl_samples/CapsBasic
#define _OCL_CHECK_ERRORS(ERR)		   \
  if(ERR != CL_SUCCESS) {		   \
    std::cout				   \
      << "OpenCL error with code " << ERR  \
      << " happened in file " << __FILE__  \
      << " at line " << __LINE__	   \
      << ". Exiting...\n";		   \
    exit(1);				   \
  }

#define _OCL_PRINT_TEXT_PROPERTY(DEVICE, NAME)		\
  {							\
    /* When we query for string properties, first we */	\
    /* need to get string length:                    */	\
    size_t property_length = 0;				\
    cl_int err = clGetDeviceInfo(			\
				 DEVICE,		\
				 NAME,			\
				 0,			\
				 0,			\
				 &property_length);	\
    _OCL_CHECK_ERRORS(err);				\
    /* Then allocate buffer. No need to add 1 symbol */	\
    /* to store terminating zero; OpenCL takes care  */	\
    /* about it:                                     */	\
    char* property_value = new char[property_length];	\
    err = clGetDeviceInfo(				\
			  DEVICE,			\
			  NAME,				\
			  property_length,		\
			  property_value,		\
			  0);				\
    _OCL_CHECK_ERRORS(err);				\
    std::cout						\
      << "    " << #NAME << ": "			\
      << property_value << std::endl;			\
    delete [] property_value;				\
  }

#define _OCL_PRINT_NUMERIC_PROPERTY(DEVICE, NAME, TYPE)		 \
  {								 \
    TYPE property_value;					 \
    size_t property_length = 0;					 \
    cl_int err = clGetDeviceInfo(				 \
				 DEVICE,			 \
				 NAME,				 \
				 sizeof(property_value),	 \
				 &property_value,		 \
				 &property_length);		 \
    assert(property_length == sizeof(property_value));		 \
    _OCL_CHECK_ERRORS(err);					 \
    std::cout							 \
      << "    " << #NAME << ": "				 \
      << property_value << std::endl;				 \
  }

// ----------------------------------------------------------------
// util-linux-2.13-pre7/schedutils/taskset.c
void get_cores(char *str)
{
  cpu_set_t mask;
  sched_getaffinity(0, sizeof(cpu_set_t), &mask);

  char *ptr = str;
  int i, j, entry_made = 0;
  for (i = 0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, &mask)) {
      int run = 0; 
      entry_made = 1;
      for (j = i + 1; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, &mask)) run++;
        else break;
      }
      if (!run)
        sprintf(ptr, "%d,", i);
      else if (run == 1) {
        sprintf(ptr, "%d,%d,", i, i + 1);
        i++;
      } else {
        sprintf(ptr, "%d-%d,", i, i + run);
        i += run;
      }
      while (*ptr != 0) ptr++;
    }
  }     
  ptr -= entry_made;
  *ptr = 0;
}       

#define MAX_UUID_STRING_SIZE    49
static char hexdigit(int i)
{
  return (i>9) ? 'a' - 10 + i : '0' + i;
}

static void generic_uuid_to_string(const uint8_t * id, int bytes, char * s)
{
  for(int i=bytes-1; i>=0; --i) {
    *s++ = hexdigit(id[i] / 0x10);
    *s++ = hexdigit(id[i] % 0x10);
    if(i >= 6 && i <= 12 && (i & 1) == 0) *s++ = '-';
  }
  *s = '\0';
}

int main( int argc, char* argv[] )
{
  // Initialize MPI

  int rnk, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm world = MPI_COMM_WORLD;

  MPI_Comm_rank(world, &rnk);
  MPI_Comm_size(world, &nprocs);

  char nname[16];
  gethostname(nname, 16);

  // Initialize GPUs

  char buf[4096];

  // OpenCL Platform
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
  _OCL_CHECK_ERRORS(err); 

  cl_platform_id * platforms = new cl_platform_id[num_platforms];
  err = clGetPlatformIDs(num_platforms, platforms, 0);
  _OCL_CHECK_ERRORS(err);
  printf("num_platforms= %i\n",num_platforms);

  int global_index = 0;
  for(int i=0; i<num_platforms; ++i) {

    err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
    _OCL_CHECK_ERRORS(err);

    const int num_device_types = 3;

    struct {
      cl_device_type type;
      const char * name;
      cl_uint count;
    } devices[] = {
      {CL_DEVICE_TYPE_CPU,         "CL_DEVICE_TYPE_CPU",         0},
      {CL_DEVICE_TYPE_GPU,         "CL_DEVICE_TYPE_GPU",         0},
      {CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", 0}
    };

    // get device counts for this platform

    for (int j=0; j<num_device_types; ++j) {
      err = clGetDeviceIDs(platforms[i], devices[j].type, 0, 0, &devices[j].count);
      if(err == CL_DEVICE_NOT_FOUND) devices[j].count = 0;
      else _OCL_CHECK_ERRORS(err);
    }

    // only test GPUs
    //for(int j=0; j<num_device_types; ++j) {
    int j = 1;

      cl_device_id device_id[256];

      err = clGetDeviceIDs(platforms[i], devices[j].type, MAX_DEVICE_SIZE, device_id, &devices[j].count);
      if(err!= CL_DEVICE_NOT_FOUND) _OCL_CHECK_ERRORS(err);

      // this is the same set printed in intel_ocl_samples/CapsBasic

      for (int k=0; k<devices[j].count; ++k) {

	size_t name_length = 0;
	err = clGetDeviceInfo(device_id[k], CL_DEVICE_NAME, 0, 0, &name_length);
	_OCL_CHECK_ERRORS(err);

	char * name = new char[name_length];
	err = clGetDeviceInfo(device_id[k], CL_DEVICE_NAME, name_length, name, 0);
	_OCL_CHECK_ERRORS(err);

	cl_uchar * uuid = new cl_uchar[CL_UUID_SIZE_KHR];
	err = clGetDeviceInfo(device_id[k], CL_DEVICE_UUID_KHR, CL_UUID_SIZE_KHR, uuid, 0);
	printf("  [%i] Platform[ %s ] Type[ %s ] Device[ %s ]  uuid= ", global_index, buf, devices[j].name, name);

        //printf("CL_UUID_SIZE_KHR= %i\n",CL_UUID_SIZE_KHR);
	printf("%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
			uuid[0],uuid[1],uuid[2],uuid[3],uuid[4],
			uuid[5],uuid[6],
			uuid[7],uuid[8],
			uuid[9],uuid[10],
			uuid[11],uuid[12],uuid[13],uuid[14],uuid[15]
			);

	global_index++;
	delete [] name;
	delete [] uuid;
      } // loop over devices of type

    //} // loop over device types
  
  }
}
