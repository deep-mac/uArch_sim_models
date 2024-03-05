/opt/nvidia/nsight-compute/2023.1.1/ncu --target-processes all --metrics gpu__time_duration.sum,gpc__cycles_elapsed.max,launch__thread_count,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size python3 test_cnn3_f.py 10 | tee ncu_reports/cnn3_f.log
/opt/nvidia/nsight-compute/2023.1.1/ncu --target-processes all --metrics gpu__time_duration.sum,gpc__cycles_elapsed.max,launch__thread_count,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size python3 test_cnn3_f_2d.py 10 | tee ncu_reports/cnn3_f_2d.log
/opt/nvidia/nsight-compute/2023.1.1/ncu --target-processes all --metrics gpu__time_duration.sum,gpc__cycles_elapsed.max,launch__thread_count,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size python3 test_cnn3_f_2d.py 10 LC | tee ncu_reports/cnn3_f_2d_LC.log