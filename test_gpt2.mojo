from std.collections import InlineArray
from std.time import perf_counter_ns
from std.sys.info import size_of
from std.memory import UnsafePointer, alloc, memcpy


from train_gpt2 import GPT2, ParameterTensors

comptime dtype = DType.float32
comptime FLOAT = SIMD[dtype, 1]

comptime dtype_int = DType.int32
comptime INT = SIMD[dtype_int, 1]

comptime FloatPtr = UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
comptime IntPtr = UnsafePointer[Scalar[dtype_int], MutUntrackedOrigin]

comptime SIZEOF_INT = size_of[DType.int32]()
comptime SIZEOF_FLOAT = size_of[DType.float32]()


# poor man's tensor checker
def check_tensor(
    a: FloatPtr,
    b: FloatPtr,
    n: Int,
    label: String,
) -> Bool:
    var print_upto: Int = 5
    var ok: Bool = True
    var maxdiff: FLOAT = 0.0
    var tol: FLOAT = 2e-2

    print(label)

    for i in range(n):
        # look at the diffence at position i of these two tensors
        var diff = abs(a[i] - b[i])

        # keep track of the overall error
        ok = ok and (diff <= tol)

        if diff > maxdiff:
            maxdiff = diff

        # for the first few elements of each tensor, pretty print
        # the actual numbers, so we can do a visual, qualitative proof/assessment
        if i < print_upto:
            if diff <= tol:
                if i < print_upto:
                    print("OK ", end="")
            else:
                if i < print_upto:
                    print("NOT OK ", end="")

            print(a[i], b[i])

    # prvar the:Int final result
    if ok:
        print("TENSOR OK")
    else:
        print("TENSOR NOT OK, maxdif =", maxdiff)
    return ok


def read_to_dtype_pointer[
    T: DType
](
    ptr: UnsafePointer[Scalar[T], MutUntrackedOrigin], file_handle: FileHandle, size: Int
) raises -> None:
    # Read directly into the pointer using read_bytes
    var bytes_to_read = size * size_of[Scalar[T]]()
    var bytes_data = file_handle.read_bytes(bytes_to_read)
    memcpy(
        dest=ptr.bitcast[UInt8](),
        src=bytes_data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        count=bytes_to_read,
    )


def main() raises:
    # build the GPT-2 model from a checkpoint
    var model = GPT2("gpt2_124M.bin")

    var C: Int = model.config.channels
    var V: Int = model.config.vocab_size
    var maxT: Int = model.config.max_seq_len
    var L: Int = model.config.num_layers

    # load additional information that we will use for debugging and error checking

    var state_file = open("gpt2_124M_debug_state.bin", "r")

    var state_header = alloc[Scalar[dtype_int]](256)
    read_to_dtype_pointer[DType.int32](state_header, state_file, 256)

    if state_header[0] != 20240327:
        raise Error(t"Bad magic state file {state_header[0]}")
    if state_header[1] != 2:
        raise Error(t"Bad version in state file {state_header[1]}")

    var B: Int = Int(state_header[2])  # batch size, e.g. 4
    var T: Int = Int(
        state_header[3]
    )  # time / sequence length (e.g. 64, up to maxT)

    print("[State]")
    print("batch_size:", B)
    print("seq_len:", T)

    var expected_grads = ParameterTensors(model.param_sizes)
    var expected_grads_memory = expected_grads.params_memory

    # inputs and expected outputs, only used for error checking

    var x = alloc[Scalar[dtype_int]](B * T)
    var y = alloc[Scalar[dtype_int]](B * T)

    var expected_logits = alloc[Scalar[dtype]](B * T * V)
    var expected_loss = alloc[Scalar[dtype]](1)

    # read reference information from Python

    read_to_dtype_pointer[DType.int32](x, state_file, B * T)
    read_to_dtype_pointer[DType.int32](y, state_file, B * T)
    read_to_dtype_pointer[DType.float32](expected_logits, state_file, B * T * V)
    read_to_dtype_pointer[DType.float32](expected_loss, state_file, 1)
    read_to_dtype_pointer[DType.float32](
        expected_grads_memory, state_file, model.num_parameters
    )

    state_file.close()

    # overall OK signal for the test
    var allok: Bool = True
    var elapsed_time_ms: Float64

    # let's do 10 training iterations, following the pytorch code

    # var losses = InlinedFixedVector[type=Float32,size=10](10)

    var expected_losses: List[Float32] = [
        5.270007133483887,
        4.059706687927246,
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564197540283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615,
    ]

    for step in range(10):
        var start = perf_counter_ns()
        model.forward(x, y, B, T)
        model.zero_grad()
        model.backward()

        elapsed_time_ms = Float64(perf_counter_ns() - start) / 1_000_000
        if step == 0:
            # error checking at step 0 for reference activations/gradients
            ref acts = model.acts.value()
            ref grads = model.grads.value()

            # at this point, target should be equal to expected_logits, let's compare

            var logits_ok: Bool = True

            for i in range(B * T * V):
                if i < 3:
                    print(expected_logits[i], acts.logits[i])

                if abs(expected_logits[i] - acts.logits[i]) >= 1e-2:
                    print("MISMATCH AT INDEX " + String(i) + ":")
                    print(expected_logits[i], acts.logits[i])
                    logits_ok = False
                    break

            if not logits_ok:
                print("NOT ", end="")
            print("OK (LOGITS)")
            allok = allok and logits_ok

            # compare the achieved loss
            if abs(model.mean_loss - expected_loss[0]) >= 1e-2:
                print("LOSS MISMATCH:", model.mean_loss, expected_loss[0])
                allok = False
            else:
                print("LOSS OK:", model.mean_loss, expected_loss[0])

            # finally check all the gradients
            var gradoks = InlineArray[Bool, 16](fill=False)

            gradoks[0] = check_tensor(
                grads.wte, expected_grads.wte, V * C, "dwte"
            )
            gradoks[1] = check_tensor(
                grads.wpe, expected_grads.wpe, maxT * C, "dwpe"
            )
            gradoks[2] = check_tensor(
                grads.ln1w, expected_grads.ln1w, L * C, "dln1w"
            )
            gradoks[3] = check_tensor(
                grads.ln1b, expected_grads.ln1b, L * C, "dln1b"
            )
            gradoks[4] = check_tensor(
                grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "dqkvw"
            )
            gradoks[5] = check_tensor(
                grads.qkvb, expected_grads.qkvb, L * 3 * C, "dqkvb"
            )
            gradoks[6] = check_tensor(
                grads.attprojw,
                expected_grads.attprojw,
                L * C * C,
                "dattprojw",
            )
            gradoks[7] = check_tensor(
                grads.attprojb,
                expected_grads.attprojb,
                L * C,
                "dattprojb",
            )
            gradoks[8] = check_tensor(
                grads.ln2w, expected_grads.ln2w, L * C, "dln2w"
            )
            gradoks[9] = check_tensor(
                grads.ln2b, expected_grads.ln2b, L * C, "dln2b"
            )
            gradoks[10] = check_tensor(
                grads.fcw, expected_grads.fcw, L * 4 * C * C, "dfcw"
            )
            gradoks[11] = check_tensor(
                grads.fcb, expected_grads.fcb, L * 4 * C, "dfcb"
            )
            gradoks[12] = check_tensor(
                grads.fcprojw,
                expected_grads.fcprojw,
                L * C * 4 * C,
                "dfcprojw",
            )
            gradoks[13] = check_tensor(
                grads.fcprojb, expected_grads.fcprojb, L * C, "dfcprojb"
            )
            gradoks[14] = check_tensor(
                grads.lnfw, expected_grads.lnfw, C, "dlnfw"
            )
            gradoks[15] = check_tensor(
                grads.lnfb, expected_grads.lnfb, C, "dlnfb"
            )

            for i in range(16):
                allok = allok and gradoks[i]

        model.update(1e-4, 0.9, 0.999, 1e-8, 0.01, step + 1)

        var expected_loss = expected_losses[step]
        var actual_loss = model.mean_loss
        var step_loss_ok = abs(expected_loss - actual_loss) < 1e-2
        allok = allok and step_loss_ok

        # prvar the:Int timing information at the end

        print(
            t"step {step}: loss {model.mean_loss} (took {elapsed_time_ms} ms) OK = {step_loss_ok}"
        )

    print("overall okay:", allok)

    # free everything
    x.free()
    y.free()
    expected_logits.free()
    expected_loss.free()
    expected_grads_memory.free()
