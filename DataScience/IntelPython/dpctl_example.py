import dpctl

dpctl.lsplatform()
num_devices = dpctl.get_num_devices(device_type="gpu")
device_list =  dpctl.get_devices(device_type="gpu")
print(f"Found {num_devices} GPU devices")
for device in device_list:
    print(f"\t{device}")
print("\nFound CPU devices: ", dpctl.has_cpu_devices())


