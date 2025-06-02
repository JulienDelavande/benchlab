ampere_sgemm_128x64_tn
Begins: 2,32255s
Ends: 2,32304s (+488,382 μs)
grid:  <<<86, 1, 2>>>
block: <<<128, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 12 800 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 122
Local Memory Per Thread: 0 bytes
Local Memory Total: 70 778 880 bytes
Shared Memory executed: 65 536 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 33,3333 %
Cluster X: 0
Cluster Y: 0
Cluster Z: 0
Max Potential Cluster Size: 0
Max Active Clusters: 0
Launched from thread: 2409
Latency: ←904,260 μs
Correlation ID: 1511
Stream: Default stream 7

ampere_sgemm_128x64_tn (86, 1, 2)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.24
    SM Frequency                    Ghz         1.32
    Elapsed Cycles                cycle       710483
    Memory Throughput                 %        73.74
    DRAM Throughput                   %        73.74
    Duration                         us       538.94
    L1/TEX Cache Throughput           %        48.42
    L2 Cache Throughput               %        27.46
    SM Active Cycles              cycle    575289.93
    Compute (SM) Throughput           %        47.61
    ----------------------- ----------- ------------

    OPT   Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the    
          DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the       
          bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or  
          whether there are values you can (re)compute.                                                                 

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    172
    Registers Per Thread             register/thread             122
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block           12.80
    # SMs                                         SM              80
    Threads                                   thread           22016
    Uses Green Context                                             0
    Waves Per SM                                                0.54
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            4
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           16
    Theoretical Occupancy                     %        33.33
    Achieved Occupancy                        %        17.55
    Achieved Active Warps Per SM           warp         8.43
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 47.34%                                                                                    
          The difference between calculated theoretical (33.3%) and measured achieved occupancy (17.6%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 66.67%                                                                                    
          The 4.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (33.3%) is limited by the number of required      
          registers. This kernel's theoretical occupancy (33.3%) is limited by the required amount of shared memory.    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle   2481405.33
    Total DRAM Elapsed Cycles        cycle     40383488
    Average L1 Active Cycles         cycle    575289.93
    Total L1 Elapsed Cycles          cycle     56349566
    Average L2 Active Cycles         cycle    643652.77
    Total L2 Elapsed Cycles          cycle     32188224
    Average SM Active Cycles         cycle    575289.93
    Total SM Elapsed Cycles          cycle     56349566
    Average SMSP Active Cycles       cycle    577349.57
    Total SMSP Elapsed Cycles        cycle    225398264
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 14.92%                                                                                          
          One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum  
          instance value is 18.27% above the average, while the minimum instance value is 8.31% below the average.      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.8%                                                                                           
          One or more SMSPs have a much higher number of active cycles than the average number of active cycles.        
          Maximum instance value is 18.06% above the average, while the minimum instance value is 8.55% below the       
          average.                                                                                                      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 14.92%                                                                                          
          One or more L1 Slices have a much higher number of active cycles than the average number of active cycles.    
          Maximum instance value is 18.27% above the average, while the minimum instance value is 8.31% below the       
          average. 