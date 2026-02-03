// Inspired from: https://github.com/NVIDIA/cutlass/pull/1413

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gemm_cuda.h"

// #include <ATen/ATen.h>
// #include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/gemm/device/gemm_universal_streamk_with_broadcast.h>

#include <cutlass/util/reference/host/error_metrics.h>
#include <cutlass/util/reference/host/tensor_foreach.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>


torch::Tensor matmul_w4a8(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &alphaCol, const torch::Tensor &alphaRow) {
    torch::checkAllSameGPU("W4A8Matmul", {{A, "A", 0}, {B, "B", 1}});
    int32_t M = A.size(0);
    int32_t N = B.size(0);
    int32_t K = A.size(1);  // 4bit packing is on the columns
    auto D = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
    // auto E = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device())); // Initialize E with zeros

    // std::cout << "E tensor created successfully." << std::endl;

    int64_t lda = A.stride(0);
    int64_t ldb = B.stride(1);
    int64_t ldc = D.stride(0);

    // A matrix configuration
    using         ElementA         = int8_t;                                    // Element type for A matrix operand
    using         LayoutA          = cutlass::layout::RowMajor;                        // Layout type for A matrix operand
    constexpr int AlignmentA       = 128 / cutlass::sizeof_bits<ElementA>::value;      // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB         = cutlass::int4b_t;                                  // Element type for B matrix operand
    using         LayoutB          = cutlass::layout::ColumnMajor;                        // Layout type for B matrix operand
    constexpr int AlignmentB       = 128 / cutlass::sizeof_bits<ElementB>::value;      // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // C1/C2/D matrix configuration
    using         ElementC         = float;        //cutlass::half_t;                   // Element type for C matrix operands
    using         LayoutC          = cutlass::layout::RowMajor;                        // Layout type for C matrix operands
    constexpr int AlignmentC       = 128 / cutlass::sizeof_bits<ElementC>::value;      // Memory access granularity/alignment of C matrices in units of elements (up to 16 bytes)

    // Output matrix configuration
    using         ElementOutput    = float;                                          // Element type for output matrix operands
    using         LayoutOutput     = cutlass::layout::RowMajor;                        // Layout type for output matrix operands
    // constexpr int AlignmentOutput  = 128 / cutlass::sizeof_bits<ElementOutput>::value; // Memory access granularity/alignment of output matrices in units of elements (up to 16 bytes)

    // Multiply-accumulate blocking/pipelining details
    using ElementAccumulator  = int32_t;                                 // Element type for internal accumulation
    using ElementCompute      = float;  //cutlass::half_t;                          // Element type for compute
    using ArchTag             = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassTensorOp;           // Operator class tag
    using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 64>;   // Threadblock-level tile size (concept: GemmShape)
    using WarpShape           = cutlass::gemm::GemmShape<64, 64, 64>;     // Warp-level tile size (concept: GemmShape)
    using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 32>;      // Instruction-level tile size (concept: GemmShape)
    constexpr int NumStages   = 4;                                        // Number of global->shared pipeline stages used in the GEMM mainloop
    constexpr int EVTEpilogueStages = 1;   

    // StreamK device GEMM implementation type with EVT
    using namespace cute;

    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, 
    WarpShape, 
    ElementC,  
    AlignmentC,  //4
    EVTEpilogueStages
    >;

    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

    // alphaCol    [M, 1]    fp32
    using V1Broadcast = cutlass::epilogue::threadblock::VisitorColBroadcast<
        OutputTileThreadMap, ElementC,
        cute::Stride<_1, _0, int32_t>  // StrideMN
    >;

    // alphaRow    [1, N]    fp32
    using V2Broadcast = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, ElementC,
        cute::Stride<_0, _1, int32_t>  // StrideMNL
    >;

    // mul
    using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest
    >;

    // alphaCol * accumulator
    using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
        Compute0,
        Accum,
        V1Broadcast>;

    // mul
    using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest
    >;

    // alphaRow * alphaCol * accumulator
    using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<
        Compute1,
        EVTCompute0,
        V2Broadcast>;

    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, _1, int64_t>
    >;

    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<
        StoreD,
        EVTCompute1>;

    // using EVTD = cutlass::epilogue::threadblock::Sm80EVT<
    // StoreD,
    // Accum
    // >;

    using EVTKernelStreamK =
        typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
        ElementC, LayoutC, AlignmentC,
        ElementAccumulator,
        ElementCompute,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EVTD,
        cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
        NumStages,
        cutlass::arch::OpMultiplyAddMixedInputUpcast,
        EVTEpilogueStages
    >::GemmKernel;

    using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversalAdapter<EVTKernelStreamK>;

    // Populates a DeviceGemmStreamK::Arguments structure from the given commandline options

    // Ensure the input tensors are in the correct device and layout
    auto tensor_a = A.contiguous();
    auto tensor_b = B.contiguous();
    auto tensor_v1 = alphaCol.contiguous();
    auto tensor_v2 = alphaRow.contiguous();
    auto tensor_d = D.contiguous();    

    //ok
    // typename EVTD::Arguments callback_args{
    //     {
    //         {}
    //     },                                                                                                               // EVTCompute1  
    //     {D.data_ptr<ElementC>(), cute::make_stride(int64_t(N), cute::_1{}, int64_t(M * N))}  // StoreD
    //                                                                    // D
    // };                                                                                                                   // EVTD

    typename EVTD::Arguments callback_args{  // EVTD
    { // EVTCompute1
      { // EVTCompute0
        {}, // Accum
        {tensor_v1.data_ptr<ElementC>(), ElementC(0), {_1{},_0{},int32_t(M)}}, // V1 Broadcast
        {}  // Compute0
      },  // EVTCompute0
      {tensor_v2.data_ptr<ElementC>(), ElementC(0), {_0{}, _1{}, int32_t(N)}}, // V2 Broadcast
      {} // Compute1                                                                                                             
    }, // EVTCompute1
    {tensor_d.data_ptr<ElementC>(), {int64_t{N}, _1{}, int64_t{M*N}}}  // D
   };  // EVTD
    

    using GemmCoord = cutlass::gemm::GemmCoord;
    // cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
    cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;  // universal mode

    int batch_count = 1;
    // Construct Gemm ProblemSize with user defined output size
    // cutlass::gemm::GemmCoord problem_size = {M, N, K};
    cutlass::gemm::GemmCoord problem_size{M, N, K};

    int64_t stride_A = M * K;
    int64_t stride_B = N * K;
    // int64_t stride_C = M * N;
    // int64_t stride_D = M * N;                                                                                               // EVTD

    // typename DeviceGemmStreamK::Arguments arguments(
    //     mode,                                     // universal mode
    //     problem_size,                             // problem_size
    //     batch_count,                              // batch count / splitk slices
    //     callback_args,                            // argument of EVT callbacks
    //     A.data_ptr<int8_t>(),            // ptr_A
    //     B.data_ptr<int8_t>(),           // ptr_B
    //     nullptr,                                  // ptr_C (unused)
    //     nullptr,                                  // ptr_D (unused)
    //     stride_A,                                 // batch_stride_A
    //     stride_B,                                 // batch_stride_B
    //     stride_C,                                        // batch_stride_C (unused)
    //     stride_D,                                        // batch_stride_D (unused)
    //     K,                       // stride_a
    //     K,                       // stride_b
    //     N,                                        // stride_c (unused)
    //     N);                                      // stride_d (unused)
                            
    std::cout << "tensor_a.stride(0)" << tensor_a.stride(0)<< std::endl;
    std::cout << "tensor_b.stride(0)." << tensor_b.stride(0)<< std::endl;
    std::cout << "tensor_b.stride(1)." << tensor_b.stride(1)<< std::endl;
    std::cout << "tensor_d.stride(0)." << tensor_d.stride(0)<< std::endl;

    typename DeviceGemmStreamK::Arguments arguments(
        mode,                                     // universal mode
        problem_size,                             // problem_size
        batch_count,                              // batch count / splitk slices
        callback_args,                            // argument of EVT callbacks
        A.data_ptr<int8_t>(),            // ptr_A
        (cutlass::int4b_t *)B.data_ptr<uint8_t>(),            // ptr_B
        nullptr,                                  // ptr_C (unused)
        nullptr,                                  // ptr_D (unused)
        stride_A,                                 // batch_stride_A
        stride_B,                                 // batch_stride_B
        0,                                        // batch_stride_C (unused)
        0,                                        // batch_stride_D (unused)
        tensor_a.stride(0),                       // stride_a
        K,                       // stride_b
        0,                                        // stride_c (unused)
        0);                                       // stride_d (unused)

    DeviceGemmStreamK gemm_op;

    // // Using the arguments, query for extra workspace required for matrix multiplication computation
    // size_t workspace_size = Gemm::get_workspace_size(arguments);

    // // Allocate workspace memory
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // // Check the problem size is supported or not 
    // cutlass::Status status = gemm_op.can_implement(arguments);
    // CUTLASS_CHECK(status);

    // // Initialize CUTLASS kernel with arguments and workspace pointer
    // status = gemm_op.initialize(arguments, workspace.get());
    // CUTLASS_CHECK(status);

    // status = gemm_op();
    // cudaDeviceSynchronize();
    // CUTLASS_CHECK(status);
  
    
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());

    // Using the arguments, query for extra workspace required for matrix
    // multiplication computation
    size_t workspace_size = DeviceGemmStreamK::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot implement");
    }

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot initialize");
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("cutlass cannot run");
    }

    return tensor_d;
}