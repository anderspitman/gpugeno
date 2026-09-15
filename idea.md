I want to create a prototype based off the pileup functionality in ./cubayes.

I want to implement pileup in a cross-platform way, ie no dependency on CUDA.

I think a way to do this would be to implement pileup with Vulkan and WebGPU,
and pair it with libdeflate on the CPU, since implementing an nvcomp
alternative in wgpu sounds like a much more difficult task.

The basic flow I'm imagining is:

1. Read chunks of BAM data from disk.
2. Identify BGZF boundaries and enqueue each bgzf block as a work item
3. Have multiple libdeflate worker threads decompress the blocks
4. Transfer the decompressed data to GPU
5. Run pileup on the data
6. Transfer the results back and display them

So what I want is a program that uses libdeflate for decompression, but where
the pileup engine can be switched between cuda, vulkan, and WebGPU, so I can
compare the performance of each.

Open questions:

* Should this be a Rust or C/C++ program? C would probably be better for cuda
  and Vulkan, but Rust has wgpu which is really nice for WebGPU. Could use
  Dawn, but I feel like it's not as popular as wgpu
