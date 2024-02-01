#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <time.h>
#include <mpi.h>

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

// xthi.c from http://docs.cray.com/books/S-2496-4101/html-S-2496-4101/cnlexamples.html

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

// from Servesh
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

void dev_uuid()
{
  // get vector of all available devices

  std::vector<sycl::device> devices = sycl::device::get_devices();
  int num_devices = devices.size();

  uint32_t num_drivers = 0;
  ze_driver_handle_t driverHandle;
  zeDriverGet(&num_drivers, &driverHandle);

  zeInit(ZE_INIT_FLAG_GPU_ONLY);

  // print useful information for each device

  int global_index = 0;
  for(int i=0; i<num_devices; ++i) {

    sycl::platform p = devices[i].get_platform();

    std::string platform_name = p.get_info<sycl::info::platform::name>();
    std::string device_name = devices[i].get_info<sycl::info::device::name>();
    int device_id = devices[i].get_info<sycl::info::device::vendor_id>();

    auto lz_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);

    ze_device_properties_t dev_prop = {};
    zeDeviceGetProperties(lz_dev, &dev_prop);

    char type[256];
    if(devices[i].is_gpu()) strcpy(type,"GPU");
    else if(devices[i].is_accelerator()) strcpy(type,"ACCELERATOR");
    else if(devices[i].is_cpu()) strcpy(type,"CPU");

    char id[MAX_UUID_STRING_SIZE];
    generic_uuid_to_string((const uint8_t *) &(dev_prop.uuid), ZE_MAX_DEVICE_UUID_SIZE, id);
/*
    printf(" [%i] Platform[%s] Type[%s] Device[%s, %i, %s]  lz uuid= %u",i,
           platform_name.c_str(), type, 
	   device_name.c_str(), device_id, id, 
	   dev_prop.subdeviceId);
*/
    printf(" %s",id);
  }

}

int main(int argc, char *argv[])
{
  // Initialize MPI
  
  int rnk, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm world = MPI_COMM_WORLD;
  
  MPI_Comm_rank(world, &rnk);
  MPI_Comm_size(world, &nprocs);

  char nname[16];
  gethostname(nname, 16);
  
  // Initialize gpu

  std::vector<sycl::device> devices = sycl::device::get_devices();
  int num_devices = devices.size();

  for(int ir=0; ir<nprocs; ++ir) {

    if(ir == rnk) {

        char list_cores[7*CPU_SETSIZE];
        get_cores(list_cores);

        printf("To affinity and beyond!! nname= %s  rnk= %d  list_cores= (%s)  num_devices= %i  gpu_uuid= ",
               nname, rnk, list_cores, num_devices);
	dev_uuid();

      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Finalize();
}
