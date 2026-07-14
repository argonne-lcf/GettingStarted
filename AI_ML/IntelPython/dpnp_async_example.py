import dpnp as np
from time import perf_counter

N = 8192
x = np.random.randn(N,N)

for i in range(10):
    y = np.matmul(x,x)

# async
times_async = []
for i in range(10):
    tic = perf_counter()
    y = np.matmul(x,x)
    toc = perf_counter()
    times_async.append(toc-tic)
print(f'Async execution time: {sum(times_async)/len(times_async):.4f} seconds (just measuring kernel launch)')

# sync
times_sync = []
for i in range(10):
    tic = perf_counter()
    y = np.matmul(x,x)
    y.sycl_queue.wait()
    toc = perf_counter()
    times_sync.append(toc-tic)
print(f'Sync execution time: {sum(times_sync)/len(times_sync):.4f} seconds (measuring kernel execution)')

