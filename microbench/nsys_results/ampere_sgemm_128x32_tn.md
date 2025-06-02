ampere_sgemm_128x32_tn
Begins: 2,64057s
Ends: 2,64153s (+960,251 μs)
grid:  <<<86, 3, 3>>>
block: <<<256, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 16 384 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 57
Local Memory Per Thread: 0 bytes
Local Memory Total: 70 778 880 bytes
Shared Memory executed: 102 400 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 66,6667 %
Cluster X: 0
Cluster Y: 0
Cluster Z: 0
Max Potential Cluster Size: 0
Max Active Clusters: 0
Launched from thread: 2409
Latency: ←10,395 μs
Correlation ID: 1790
Stream: Default stream 7

ampere_sgemm_128x32_tn (86, 3, 3)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.24
    SM Frequency                    Ghz         1.32
    Elapsed Cycles                cycle      1397018
    Memory Throughput                 %        79.39
    DRAM Throughput                   %        79.39
    Duration                         ms         1.06
    L1/TEX Cache Throughput           %        56.51
    L2 Cache Throughput               %        50.83
    SM Active Cycles              cycle   1389859.62
    Compute (SM) Throughput           %        49.13
    ----------------------- ----------- ------------

    OPT   Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the    
          DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the       
          bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or  
          whether there are values you can (re)compute.                                                                 

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    774
    Registers Per Thread             register/thread              57
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block           16.38
    # SMs                                         SM              80
    Threads                                   thread          198144
    Uses Green Context                                             0
    Waves Per SM                                                2.42
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            5
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        62.21
    Achieved Active Warps Per SM           warp        29.86
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 33.33%                                                                                    
          The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of required      
          registers.                                                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle   5251561.33
    Total DRAM Elapsed Cycles        cycle     79381504
    Average L1 Active Cycles         cycle   1389859.62
    Total L1 Elapsed Cycles          cycle    114305794
    Average L2 Active Cycles         cycle   1366562.67
    Total L2 Elapsed Cycles          cycle     63274032
    Average SM Active Cycles         cycle   1389859.62
    Total SM Elapsed Cycles          cycle    114305794
    Average SMSP Active Cycles       cycle   1378629.52
    Total SMSP Elapsed Cycles        cycle    457223176
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 5.513%                                                                                          
          One or more SMs have a much higher number of active cycles than the average number of active cycles.          
          Additionally, other SMs have a much lower number of active cycles than the average number of active cycles.   
          Maximum instance value is 5.67% above the average, while the minimum instance value is 5.46% below the        
          average.                                                                                                      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.827%                                                                                          
          One or more SMSPs have a much higher number of active cycles than the average number of active cycles.        
          Additionally, other SMSPs have a much lower number of active cycles than the average number of active         
          cycles. Maximum instance value is 6.04% above the average, while the minimum instance value is 5.75% below    
          the average.                                                                                                  
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.513%                                                                                          
          One or more L1 Slices have a much higher number of active cycles than the average number of active cycles.    
          Additionally, other L1 Slices have a much lower number of active cycles than the average number of active     
          cycles. Maximum instance value is 5.67% above the average, while the minimum instance value is 5.46% below    
          the average. 