include("parallel_go_fast.jl")

using CUDA
using Test
using BenchmarkTools

N = 2^20

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
# d means "device," in contrast with "host"
y_d .+= x_d
println("broadcasted GPU add: ", @test all(Array(y_d) .== 3.0f0))
# benchmarking
function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end
@btime add_broadcast!($y_d, $x_d)

# actually use a self-implemented CUDA kernel
function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
println("sequential GPU add: ", @test all(Array(y_d) .== 3.0f0))

# BENCHMARKING
function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end
@btime bench_gpu1!($y_d, $x_d)

#PROFILING
# display(CUDA.@profile trace=true bench_gpu1!(y_d, x_d))
# turns out gpu_add1 actually runs sequentially on the GPU


# Writing a parallel GPU kernel
function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads = 256 gpu_add2!(y_d, x_d)
println("parallel GPU add: ", @test all(Array(y_d) .== 3.0f0))
# BENCHMARKING
function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads = 256 gpu_add2!(y, x)
    end
end
@btime bench_gpu2!($y_d, $x_d)

# EVEN MORE PARALLEL
function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

numblocks = ceil(Int, N / 256)

fill!(y_d, 2)
@cuda threads = 256 blocks = numblocks gpu_add3!(y_d, x_d)
println("Multiple blocks: ", @test all(Array(y_d) .== 3.0f0))

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y) / 256)
    CUDA.@sync begin
        @cuda threads = 256 blocks = numblocks gpu_add3!(y, x)
    end
end
@btime bench_gpu3!($y_d, $x_d)
# display(CUDA.@profile trace = true bench_gpu3!(y_d, x_d))

# AUTOMATICALLY CHOOSING OF THREADS AND BLOCKS
kernel = @cuda launch = false gpu_add3!(y_d, x_d)
config = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks = cld(N, threads)

fill!(y_d, 2)
kernel(y_d, x_d; threads, blocks)
println("Automatic threads & blocks: ", @test all(Array(y_d) .== 3.0f0))

function bench_gpu4!(y, x)
    kernel = @cuda launch = false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end
@btime bench_gpu4!($y_d, $x_d)
display(CUDA.@profile trace = true bench_gpu4!(y_d, x_d))