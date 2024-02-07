using Test
using BenchmarkTools

N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

# CPU - SERIAL
# y .+= x             # increment each element of y with the corresponding element of x
# @test all(y .== 3.0f0)

# CPU - PARALLEL
# this is a sequential "kernel"
function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
println("CPU sequential add: ", @test all(y .== 3.0f0))
display(@btime sequential_add!($y, $x))

# this is a parallel kernel
function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
println("CPU parallel add: ", @test all(y .== 3.0f0))

# BENCHMARK
display(@btime parallel_add!($y, $x))