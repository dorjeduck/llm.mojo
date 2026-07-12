from std.algorithm import vectorize, parallelize
from std.collections import InlineArray
from std.math import sqrt, exp, tanh, cosh, log
from std.memory import UnsafePointer, alloc, memset, memset_zero, memcpy
from std.os.path import getsize
from std.sys import CompilationTarget
from std.sys.info import simd_width_of, size_of
from std.time import perf_counter_ns


def get_simd_width() -> Int:
    if CompilationTarget.is_apple_silicon():
        return 4 * simd_width_of[dtype]()
    else:
        return 2 * simd_width_of[dtype]()


comptime dtype = DType.float32  # must be float32 for now
comptime dtype_int = DType.int32  # must be int32 for now

comptime SIZEOF_FLOAT = size_of[dtype]()
comptime SIZEOF_INT = size_of[dtype_int]()

comptime SIMD_WIDTH = get_simd_width()

comptime RU32_HEX = 0x2545F4914F6CDD1D
comptime RF32_DIV = 16777216.0


comptime FLOAT = SIMD[dtype, 1]
comptime INT = SIMD[dtype_int, 1]

comptime FloatPtr = UnsafePointer[Scalar[dtype], MutUntrackedOrigin]
comptime IntPtr = UnsafePointer[Scalar[dtype_int], MutUntrackedOrigin]

comptime M_PI: FLOAT = 3.141592653589793115997963468544185161590576171875

comptime GPT2_EOT = 50256

comptime NUM_PARALLELIZE = 8
comptime UNROLL_FACTOR = 4


## ----------------------------------------------------------------------------
# all the individual layers' forward and backward passes


def encoder_forward(
    vout: FloatPtr,
    inp: IntPtr,
    wte: FloatPtr,
    wpe: FloatPtr,
    B: Int,
    T: Int,
    C: Int,
):
    @parameter
    def _calc(b: Int):
        for t in range(T):
            # seek to the output position in out[b,t,:]
            var out_bt: FloatPtr = vout + b * T * C + t * C
            # get the index of the token at inp[b, t]
            var ix = inp[b * T + t]
            # seek to the position in wte corresponding to the token
            var wte_ix: FloatPtr = wte + Int(ix) * C
            # seek to the position in wpe corresponding to the position
            var wpe_t: FloatPtr = wpe + t * C
            # add the two vectors and store the result in out[b,t,:]

            def _op[width: Int](iv: Int) {out_bt, wte_ix, wpe_t}:
                out_bt.store(
                    iv,
                    wte_ix.load[width=width](iv) + wpe_t.load[width=width](iv),
                )

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op)

    parallelize[_calc](B)


def encoder_backward(
    dwte: FloatPtr,
    dwpe: FloatPtr,
    dout: FloatPtr,
    inp: IntPtr,
    B: Int,
    T: Int,
    C: Int,
):
    @parameter
    def _calc(b: Int):
        for t in range(T):
            var dout_bt: FloatPtr = dout + b * T * C + t * C
            var ix = inp[b * T + t]
            var dwte_ix: FloatPtr = dwte + Int(ix) * C
            var dwpe_t: FloatPtr = dwpe + t * C

            def _op[width: Int](iv: Int) {dout_bt, dwte_ix, dwpe_t}:
                var d = dout_bt.load[width=width](iv)
                dwte_ix.store(iv, dwte_ix.load[width=width](iv) + d)
                dwpe_t.store(iv, dwpe_t.load[width=width](iv) + d)

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op)

    parallelize[_calc](B)


def layernorm_forward(
    vout: FloatPtr,
    mean: FloatPtr,
    rstd: FloatPtr,
    inp: FloatPtr,
    weight: FloatPtr,
    bias: FloatPtr,
    B: Int,
    T: Int,
    C: Int,
):
    var eps: FLOAT = 1e-5

    @parameter
    def _calc(b: Int):
        for t in range(T):
            # seek to the input position inp[b,t,:]
            var x: FloatPtr = inp + b * T * C + t * C
            # calculate the mean
            var m: FLOAT = 0.0

            def _op[width: Int](iv: Int) {x, mut m}:
                m += x.load[width=width](iv).reduce_add[1]()

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op)

            m = m / FLOAT(C)

            # calculate the variance (without any bias correction)
            var v: FLOAT = 0.0

            def _op2[width: Int](iv: Int) {x, m, mut v}:
                var xshift = x.load[width=width](iv) - m
                v += pow(xshift, 2).reduce_add[1]()

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op2)

            v = v / FLOAT(C)

            # calculate the rstd
            var s: FLOAT = FLOAT(1.0) / sqrt(v + eps)

            # seek to the output position in out[b,t,:]
            var out_bt: FloatPtr = vout + b * T * C + t * C

            def _op3[width: Int](iv: Int) {s, x, m, weight, bias, out_bt}:
                var n = s * (x.load[width=width](iv) - m)  # normalized output
                out_bt.store(
                    iv,
                    n * weight.load[width=width](iv)
                    + bias.load[width=width](iv),
                )  # scale and shift it

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op3)

            # cache the mean and rstd for the backward pass later
            mean[b * T + t] = m
            rstd[b * T + t] = s

    parallelize[_calc](B)


def layernorm_backward(
    dinp: FloatPtr,
    dweight: FloatPtr,
    dbias: FloatPtr,
    dout: FloatPtr,
    inp: FloatPtr,
    weight: FloatPtr,
    mean: FloatPtr,
    rstd: FloatPtr,
    B: Int,
    T: Int,
    C: Int,
):
    @parameter
    def _calc(b: Int):
        for t in range(T):
            var dout_bt: FloatPtr = dout + b * T * C + t * C
            var inp_bt: FloatPtr = inp + b * T * C + t * C
            var dinp_bt: FloatPtr = dinp + b * T * C + t * C
            var mean_bt: FLOAT = mean[b * T + t]
            var rstd_bt: FLOAT = rstd[b * T + t]

            # first: two reduce operations
            var dnorm_mean: FLOAT = 0.0
            var dnorm_norm_mean: FLOAT = 0.0

            def _op[width: Int](iv: Int) {inp_bt, mean_bt, rstd_bt, weight, dout_bt, mut dnorm_mean, mut dnorm_norm_mean}:
                var norm_bti = (
                    inp_bt.load[width=width](iv) - mean_bt
                ) * rstd_bt
                var dnorm_i = weight.load[width=width](iv) * dout_bt.load[
                    width=width
                ](iv)
                dnorm_mean += dnorm_i.reduce_add[1]()
                dnorm_norm_mean += (dnorm_i * norm_bti).reduce_add[1]()

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op)

            dnorm_mean = dnorm_mean / FLOAT(C)
            dnorm_norm_mean = dnorm_norm_mean / FLOAT(C)

            # now iterate again and accumulate all the gradients

            def _op2[width: Int](iv: Int) {inp_bt, mean_bt, rstd_bt, weight, dout_bt, dbias, dweight, dinp_bt, dnorm_mean, dnorm_norm_mean}:
                var norm_bti = (
                    inp_bt.load[width=width](iv) - mean_bt
                ) * rstd_bt
                var dnorm_i = weight.load[width=width](iv) * dout_bt.load[
                    width=width
                ](iv)
                # gradient contribution to bias
                dbias.store(
                    iv,
                    dbias.load[width=width](iv) + dout_bt.load[width=width](iv),
                )
                # gradient contribution to weight
                dweight.store(
                    iv,
                    dweight.load[width=width](iv)
                    + norm_bti * dout_bt.load[width=width](iv),
                )
                # gradient contribution to input
                dinp_bt.store(
                    iv,
                    dinp_bt.load[width=width](iv)
                    + (dnorm_i - dnorm_mean - (norm_bti * dnorm_norm_mean))
                    * rstd_bt,
                )

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op2)

    parallelize[_calc](B)


def matmul_forward(
    vout: FloatPtr,
    inp: FloatPtr,
    weight: FloatPtr,
    bias: Optional[FloatPtr],
    B: Int,
    T: Int,
    C: Int,
    OC: Int,
):
    # most of the running time is spent here and in matmul_backward
    # OC is short for "output channels"
    # inp is (B,T,C), weight is (OC, C), bias is (OC)
    # out will be (B,T,OC)
    # pragma omp parallel for collapse(2)

    @parameter
    def _calc(b: Int):
        for t in range(T):
            var out_bt: FloatPtr = (
                vout + b * T * OC + t * OC
            )
            var inp_bt: FloatPtr = inp + b * T * C + t * C

            for o in range(OC):
                var val: FLOAT = 0.0
                if bias:
                    val = bias.value()[o]
                var wrow: FloatPtr = weight + o * C

                def _op[width: Int](iv: Int) {inp_bt, wrow, mut val}:
                    var t = inp_bt.load[width=width](iv) * wrow.load[
                        width=width
                    ](iv)
                    val += t.reduce_add[1]()

                vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op)

                out_bt[o] = val

    parallelize[_calc](B)


def matmul_backward(
    dinp: FloatPtr,
    dweight: FloatPtr,
    dbias: Optional[FloatPtr],
    dout: FloatPtr,
    inp: FloatPtr,
    weight: FloatPtr,
    B: Int,
    T: Int,
    C: Int,
    OC: Int,
):
    # most of the running time is spent here and in matmul_forward
    # this backward could be done in a single "round" of loops
    # but that doesn't afford an efficient parallelization strategy

    # backward into inp first, parallelize over B,T
    # pragma omp parallel for collapse(2)

    @parameter
    def _calc(b: Int):
        for t in range(T):
            var dout_bt: FloatPtr = (
                dout + b * T * OC + t * OC
            )
            var dinp_bt: FloatPtr = dinp + b * T * C + t * C
            for o in range(OC):
                var wrow: FloatPtr = weight + o * C
                var d: FLOAT = dout_bt[o]

                def _op[width: Int](iv: Int) {dinp_bt, wrow, d}:
                    dinp_bt.store(
                        iv,
                        dinp_bt.load[width=width](iv)
                        + wrow.load[width=width](iv) * d,
                    )  # scale and shift it

                vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op)

    parallelize[_calc](B)
    # backward into weight/bias, parallelize over output channels OC
    # pragma omp parallel for

    @parameter
    def _calc2(o: Int):
        for b in range(B):
            for t in range(T):
                var dout_bt: FloatPtr = (
                    dout + b * T * OC + t * OC
                )
                var inp_bt: FloatPtr = (
                    inp + b * T * C + t * C
                )
                var dwrow: FloatPtr = dweight + o * C
                var d: FLOAT = dout_bt[o]
                if dbias:
                    dbias.value()[o] += d

                def _op[width: Int](iv: Int) {dwrow, inp_bt, d}:
                    dwrow.store(
                        iv,
                        dwrow.load[width=width](iv)
                        + inp_bt.load[width=width](iv) * d,
                    )  # scale and shift it

                vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](C, _op)

    parallelize[_calc2](OC)


def attention_forward(
    vout: FloatPtr,
    preatt: FloatPtr,
    att: FloatPtr,
    inp: FloatPtr,
    B: Int,
    T: Int,
    C: Int,
    NH: Int,
):
    # input is (B, T, 3C) Q,K,V
    # preatt, att are (B, NH, T, T)
    # output is (B, T, C)
    var C3: Int = C * 3
    var hs: Int = C // NH  # head size
    var scale: FLOAT = FLOAT(1.0) / sqrt(FLOAT(hs))

    # pragma omp parallel for collapse(3)
    @parameter
    def _calc(b: Int):
        # for b in range(B):
        for t in range(T):
            for h in range(NH):
                var query_t: FloatPtr = (
                    inp + b * T * C3 + t * C3 + h * hs
                )
                var preatt_bth: FloatPtr = (
                    preatt + b * NH * T * T + h * T * T + t * T
                )
                var att_bth: FloatPtr = (
                    att + b * NH * T * T + h * T * T + t * T
                )

                # pass 1: calculate query dot key and maxval
                var maxval: FLOAT = FLOAT.MIN_FINITE

                for t2 in range(t + 1):
                    var key_t2: FloatPtr = (
                        inp + b * T * C3 + t2 * C3 + h * hs + C
                    )  # +C because it's key

                    # (query_t) dot (key_t2)
                    var val: FLOAT = 0.0

                    def _op[width: Int](iv: Int) {query_t, key_t2, mut val}:
                        var t = query_t.load[width=width](iv) * key_t2.load[
                            width=width
                        ](iv)
                        val += t.reduce_add[1]()

                    vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](hs, _op)

                    val *= scale
                    if val > maxval:
                        maxval = val

                    preatt_bth[t2] = val

                # pass 2: calculate the exp and keep track of sum
                var expsum: FLOAT = 0.0

                def _op2[width: Int](iv: Int) {preatt_bth, maxval, att_bth, mut expsum}:
                    var expv = exp(preatt_bth.load[width=width](iv) - maxval)
                    expsum += expv.reduce_add[1]()
                    att_bth.store(iv, expv)

                vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](t + 1, _op2)

                var expsum_inv: FLOAT = 0.0
                if expsum != 0.0:
                    expsum_inv = FLOAT(1.0) / expsum

                # pass 3: normalize to get the softmax

                def _op3[width: Int](t2: Int) {att_bth, expsum_inv}:
                    att_bth.store(
                        t2, att_bth.load[width=width](t2) * expsum_inv
                    )

                vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](t + 1, _op3)
                memset_zero(att_bth + t + 1, T - t - 1)

                # pass 4: accumulate weighted values into the output of attention
                var out_bth: FloatPtr = (
                    vout + b * T * C + t * C + h * hs
                )
                # for i in range(hs):
                #    out_bth[i] = 0.0
                memset_zero(out_bth, hs)

                for t2 in range(t + 1):
                    var value_t2 = (
                        inp + b * T * C3 + t2 * C3 + h * hs + C * 2
                    )  # +C*2 because it's value
                    var att_btht2: FLOAT = att_bth[t2]

                    def _op4[width: Int](iv: Int) {out_bth, att_btht2, value_t2}:
                        out_bth.store(
                            iv,
                            out_bth.load[width=width](iv)
                            + att_btht2 * value_t2.load[width=width](iv),
                        )

                    vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](hs, _op4)

    parallelize[_calc](B)


def attention_backward(
    dinp: FloatPtr,
    dpreatt: FloatPtr,
    datt: FloatPtr,
    dout: FloatPtr,
    inp: FloatPtr,
    att: FloatPtr,
    B: Int,
    T: Int,
    C: Int,
    NH: Int,
):
    # inp/dinp are (B, T, 3C) Q,K,V
    # att/datt/dpreatt are (B, NH, T, T)
    # dout is (B, T, C)
    var C3: Int = C * 3
    var hs: Int = C // NH  # head size
    var scale: FLOAT = FLOAT(1.0) / sqrt(FLOAT(hs))

    @parameter
    def _calc(b: Int):
        for t in range(T):
            for h in range(NH):
                var att_bth: FloatPtr = (
                    att + b * NH * T * T + h * T * T + t * T
                )
                var datt_bth: FloatPtr = (
                    datt + b * NH * T * T + h * T * T + t * T
                )
                var dpreatt_bth: FloatPtr = (
                    dpreatt + b * NH * T * T + h * T * T + t * T
                )
                var dquery_t: FloatPtr = (
                    dinp + b * T * C3 + t * C3 + h * hs
                )
                var query_t: FloatPtr = (
                    inp + b * T * C3 + t * C3 + h * hs
                )

                # backward pass 4, through the value accumulation
                var dout_bth: FloatPtr = (
                    dout + b * T * C + t * C + h * hs
                )
                for t2 in range(t + 1):
                    var value_t2: FloatPtr = (
                        inp + b * T * C3 + t2 * C3 + h * hs + C * 2
                    )  # +C*2 because it's value
                    var dvalue_t2: FloatPtr = (
                        dinp + b * T * C3 + t2 * C3 + h * hs + C * 2
                    )

                    def _op[width: Int](iv: Int) {datt_bth, value_t2, dout_bth, dvalue_t2, att_bth, t2}:
                        # for i in range(hs):
                        # in the forward pass this was:
                        # out_bth[i] += att_bth[t2] * value_t2[i]
                        # so now we have:
                        datt_bth[t2] += (
                            value_t2.load[width=width](iv)
                            * dout_bth.load[width=width](iv)
                        ).reduce_add[1]()
                        dvalue_t2.store(
                            iv,
                            dvalue_t2.load[width=width](iv)
                            + att_bth[t2] * dout_bth.load[width=width](iv),
                        )

                    vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](hs, _op)

                # backward pass 2 & 3, the softmax
                # note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in range(t + 1):

                    def _op3[width: Int](t3: Int) {att_bth, t2, dpreatt_bth, datt_bth}:
                        var local_derivative = -att_bth[t2] * att_bth.load[
                            width=width
                        ](t3)
                        dpreatt_bth.store(
                            t3,
                            dpreatt_bth.load[width=width](t3)
                            + local_derivative * datt_bth[t2],
                        )

                    vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](t + 1, _op3)
                    dpreatt_bth[t2] += att_bth[t2] * datt_bth[t2]

                # backward pass 1, the query @ key matmul
                for t2 in range(t + 1):
                    var key_t2: FloatPtr = (
                        inp + b * T * C3 + t2 * C3 + h * hs + C
                    )  # +C because it's key
                    var dkey_t2: FloatPtr = (
                        dinp + b * T * C3 + t2 * C3 + h * hs + C
                    )  # +C because it's key

                    def _op2[width: Int](iv: Int) {dquery_t, key_t2, dpreatt_bth, t2, scale, dkey_t2, query_t}:
                        # for i in range(hs):
                        # in the forward pass this was:
                        # preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale
                        # so now we have:
                        dquery_t.store(
                            iv,
                            dquery_t.load[width=width](iv)
                            + key_t2.load[width=width](iv)
                            * dpreatt_bth[t2]
                            * scale,
                        )
                        dkey_t2.store(
                            iv,
                            dkey_t2.load[width=width](iv)
                            + query_t.load[width=width](iv)
                            * dpreatt_bth[t2]
                            * scale,
                        )

                    vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](hs, _op2)

    parallelize[_calc](B)


def gelu_forward(
    vout: FloatPtr,
    inp: FloatPtr,
    N: Int,
):
    var s: FLOAT = sqrt(2.0 / M_PI)

    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    def _calc(ip: Int):
        def _op[width: Int](_iv: Int) {ip, num_vectorize, inp, s, vout}:
            var iv = ip * num_vectorize + _iv

            var x = inp.load[width=width](iv)
            var cube = 0.044715 * pow(x, 3)
            vout.store(iv, 0.5 * x * (1.0 + tanh(s * (x + cube))))

        vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize, _op)

    parallelize[_calc](NUM_PARALLELIZE)


def gelu_backward(
    dinp: FloatPtr,
    inp: FloatPtr,
    dout: FloatPtr,
    N: Int,
):
    var s: FLOAT = sqrt(2.0 / M_PI)
    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    def _calc(ip: Int):
        def _op[width: Int](_iv: Int) {ip, num_vectorize, inp, s, dinp, dout}:
            var iv = ip * num_vectorize + _iv

            var x = inp.load[width=width](iv)
            var cube = 0.044715 * pow(x, 3)
            var tanh_arg = s * (x + cube)
            var tanh_out = tanh(tanh_arg)
            var coshf_out = cosh(tanh_arg)
            var sech_out = FLOAT(1.0) / (coshf_out * coshf_out)
            var local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * s * (
                1.0 + 3.0 * 0.044715 * x * x
            )
            dinp.store(
                iv,
                dinp.load[width=width](iv)
                + local_grad * dout.load[width=width](iv),
            )

        vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize, _op)

    parallelize[_calc](NUM_PARALLELIZE)


def residual_forward(
    vout: FloatPtr,
    inp1: FloatPtr,
    inp2: FloatPtr,
    N: Int,
):
    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    def _calc(ip: Int):
        def _op[width: Int](_iv: Int) {ip, num_vectorize, vout, inp1, inp2}:
            var iv = ip * num_vectorize + _iv
            vout.store(
                iv, inp1.load[width=width](iv) + inp2.load[width=width](iv)
            )  # scale and shift it

        vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize, _op)

    parallelize[_calc](NUM_PARALLELIZE)


def residual_backward(
    dinp1: FloatPtr,
    dinp2: FloatPtr,
    dout: FloatPtr,
    N: Int,
):
    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    def _calc(ip: Int):
        def _op[width: Int](_iv: Int) {ip, num_vectorize, dinp1, dinp2, dout}:
            var iv = ip * num_vectorize + _iv

            dinp1.store(
                iv, dinp1.load[width=width](iv) + dout.load[width=width](iv)
            )  # scale and shift it
            dinp2.store(
                iv, dinp2.load[width=width](iv) + dout.load[width=width](iv)
            )  # scale and shift it

        vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize, _op)

    parallelize[_calc](NUM_PARALLELIZE)


def softmax_forward(
    probs: FloatPtr,
    logits: FloatPtr,
    B: Int,
    T: Int,
    V: Int,
    Vp: Int,
):
    # output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    # input: logits is (B,T,Vp) of the unnormalized log probabilities
    # Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    # example: Vp is 50304 and V is 50257

    @parameter
    def _calc(b: Int):
        # for b in range(B):
        for t in range(T):
            # probs <- softmax(logits)
            var logits_bt: FloatPtr = (
                logits + b * T * Vp + t * Vp
            )
            var probs_bt: FloatPtr = (
                probs + b * T * Vp + t * Vp
            )

            var maxval: FLOAT = FLOAT.MIN_FINITE
            for i in range(V):
                if logits_bt[i] > maxval:
                    maxval = logits_bt[i]

            var sum: FLOAT = 0.0

            def _op[width: Int](iv: Int) {probs_bt, logits_bt, maxval, mut sum}:
                probs_bt.store(
                    iv, exp(logits_bt.load[width=width](iv) - maxval)
                )
                sum += probs_bt.load[width=width](iv).reduce_add[1]()

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](V, _op)

            def _op2[width: Int](iv: Int) {probs_bt, sum}:
                probs_bt.store(
                    iv, probs_bt.load[width=width](iv) / sum
                )  # scale and shift it

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](V, _op2)

            # for extra super safety we may wish to include this too,
            # forcing the probabilities here to be zero, but it shouldn't matter

            memset_zero(probs_bt + V, Vp - V)

    parallelize[_calc](B)


def crossentropy_forward(
    losses: FloatPtr,
    probs: FloatPtr,
    targets: IntPtr,
    B: Int,
    T: Int,
    Vp: Int,
):
    # output: losses is (B,T) of the individual losses at each position
    # input: probs are (B,T,Vp) of the probabilities
    # input: targets is (B,T) of integers giving the correct index in logits

    @parameter
    def _calc(b: Int):
        for t in range(T):  # todo
            # loss = -log(probs[target])
            var probs_bt: FloatPtr = (
                probs + b * T * Vp + t * Vp
            )
            var ix = targets[b * T + t]
            losses[b * T + t] = -log(probs_bt.load(ix))

    parallelize[_calc](B)


def crossentropy_softmax_backward(
    dlogits: FloatPtr,
    dlosses: FloatPtr,
    probs: FloatPtr,
    targets: IntPtr,
    B: Int,
    T: Int,
    V: Int,
    Vp: Int,
):
    # backwards through both softmax and crossentropy

    @parameter
    def _calc(b: Int):
        for t in range(T):
            var dlogits_bt: FloatPtr = (
                dlogits + b * T * Vp + t * Vp
            )
            var probs_bt: FloatPtr = (
                probs + b * T * Vp + t * Vp
            )
            var dloss: FLOAT = dlosses[b * T + t]
            var ix = targets[b * T + t]

            def _op[width: Int](iv: Int) {dlogits_bt, probs_bt, dloss}:
                dlogits_bt.store(
                    iv,
                    dlogits_bt.load[width=width](iv)
                    + probs_bt.load[width=width](iv) * dloss,
                )

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](V, _op)

            if ix >= 0 and ix < INT(V):
                dlogits_bt.store(ix, dlogits_bt.load(ix) - dloss)

    parallelize[_calc](B)


# ----------------------------------------------------------------------------
# GPT-2 model definition

# the parameters of the model

comptime NUM_PARAMETER_TENSORS = 16


struct ParameterTensors(Copyable, Movable):
    var params_memory: FloatPtr

    var wte: FloatPtr  # (V, C)
    var wpe: FloatPtr  # (maxT, C)
    var ln1w: FloatPtr  # (L, C)
    var ln1b: FloatPtr  # (L, C)
    var qkvw: FloatPtr  # (L, 3*C, C)
    var qkvb: FloatPtr  # (L, 3*C)
    var attprojw: FloatPtr  # (L, C, C)
    var attprojb: FloatPtr  # (L, C)
    var ln2w: FloatPtr  # (L, C)
    var ln2b: FloatPtr  # (L, C)
    var fcw: FloatPtr  # (L, 4*C, C)
    var fcb: FloatPtr  # (L, 4*C)
    var fcprojw: FloatPtr  # (L, C, 4*C)
    var fcprojb: FloatPtr  # (L, C)
    var lnfw: FloatPtr  # (C)
    var lnfb: FloatPtr  # (C)

    def __init__(
        out self,
        param_sizes: InlineArray[Int, NUM_PARAMETER_TENSORS],
    ):
        var num_parameters: Int = 0

        for i in range(NUM_PARAMETER_TENSORS):
            num_parameters += param_sizes[i]

        # malloc all parameters all at once
        self.params_memory = alloc[Scalar[dtype]](num_parameters)

        # assign all the tensors
        var it: FloatPtr = self.params_memory
        self.wte = it
        it += param_sizes[0]
        self.wpe = it
        it += param_sizes[1]
        self.ln1w = it
        it += param_sizes[2]
        self.ln1b = it
        it += param_sizes[3]
        self.qkvw = it
        it += param_sizes[4]
        self.qkvb = it
        it += param_sizes[5]
        self.attprojw = it
        it += param_sizes[6]
        self.attprojb = it
        it += param_sizes[7]
        self.ln2w = it
        it += param_sizes[8]
        self.ln2b = it
        it += param_sizes[9]
        self.fcw = it
        it += param_sizes[10]
        self.fcb = it
        it += param_sizes[11]
        self.fcprojw = it
        it += param_sizes[12]
        self.fcprojb = it
        it += param_sizes[13]
        self.lnfw = it
        it += param_sizes[14]
        self.lnfb = it


comptime NUM_ACTIVATION_TENSORS = 23


struct ActivationTensors(Copyable, Movable):
    var acts_memory: FloatPtr

    var encoded: FloatPtr  # (B, T, C)
    var ln1: FloatPtr  # (L, B, T, C)
    var ln1_mean: FloatPtr  # (L, B, T)
    var ln1_rstd: FloatPtr  # (L, B, T)
    var qkv: FloatPtr  # (L, B, T, 3*C)
    var atty: FloatPtr  # (L, B, T, C)
    var preatt: FloatPtr  # (L, B, NH, T, T)
    var att: FloatPtr  # (L, B, NH, T, T)
    var attproj: FloatPtr  # (L, B, T, C)
    var residual2: FloatPtr  # (L, B, T, C)
    var ln2: FloatPtr  # (L, B, T, C)
    var ln2_mean: FloatPtr  # (L, B, T)
    var ln2_rstd: FloatPtr  # (L, B, T)
    var fch: FloatPtr  # (L, B, T, 4*C)
    var fch_gelu: FloatPtr  # (L, B, T, 4*C)
    var fcproj: FloatPtr  # (L, B, T, C)
    var residual3: FloatPtr  # (L, B, T, C)
    var lnf: FloatPtr  # (B, T, C)
    var lnf_mean: FloatPtr  # (B, T)
    var lnf_rstd: FloatPtr  # (B, T)
    var logits: FloatPtr  # (B, T, V)
    var probs: FloatPtr  # (B, T, V)
    var losses: FloatPtr  # (B, T)

    def __init__(
        out self,
        act_sizes: InlineArray[Int, NUM_ACTIVATION_TENSORS],
    ):
        var num_activations: Int = 0

        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += act_sizes[i]

        # malloc all activations all at once
        self.acts_memory = alloc[Scalar[dtype]](num_activations)

        # assign all the tensors
        var it: FloatPtr = self.acts_memory
        self.encoded = it
        it += act_sizes[0]
        self.ln1 = it
        it += act_sizes[1]
        self.ln1_mean = it
        it += act_sizes[2]
        self.ln1_rstd = it
        it += act_sizes[3]
        self.qkv = it
        it += act_sizes[4]
        self.atty = it
        it += act_sizes[5]
        self.preatt = it
        it += act_sizes[6]
        self.att = it
        it += act_sizes[7]
        self.attproj = it
        it += act_sizes[8]
        self.residual2 = it
        it += act_sizes[9]
        self.ln2 = it
        it += act_sizes[10]
        self.ln2_mean = it
        it += act_sizes[11]
        self.ln2_rstd = it
        it += act_sizes[12]
        self.fch = it
        it += act_sizes[13]
        self.fch_gelu = it
        it += act_sizes[14]
        self.fcproj = it
        it += act_sizes[15]
        self.residual3 = it
        it += act_sizes[16]
        self.lnf = it
        it += act_sizes[17]
        self.lnf_mean = it
        it += act_sizes[18]
        self.lnf_rstd = it
        it += act_sizes[19]
        self.logits = it
        it += act_sizes[20]
        self.probs = it
        it += act_sizes[21]
        self.losses = it


@fieldwise_init
struct GPT2Config:
    var max_seq_len: Int  # max sequence length, e.g. 1024
    var vocab_size: Int  # vocab size, e.g. 50257
    var num_layers: Int  # number of layers, e.g. 12
    var num_heads: Int  # number of heads in attention, e.g. 12
    var channels: Int  # number of channels, e.g. 768
    var padded_vocab_size: Int  # padded to e.g. %128==0, 50304


struct GPT2:
    var config: GPT2Config
    # the weights of the model, and their sizes
    var params: ParameterTensors
    var param_sizes: InlineArray[Int, NUM_PARAMETER_TENSORS]
    var params_memory: FloatPtr
    var num_parameters: Int
    # gradients of the weights (lazily allocated in backward)
    var grads: Optional[ParameterTensors]
    # buffers for the AdamW optimizer (lazily allocated in update)
    var m_memory: Optional[FloatPtr]
    var v_memory: Optional[FloatPtr]
    # the activations of the model, and their sizes (lazily allocated in forward)
    var acts: Optional[ActivationTensors]
    var act_sizes: InlineArray[Int, NUM_ACTIVATION_TENSORS]
    var num_activations: Int
    # gradients of the activations (lazily allocated in backward)
    var grads_acts: Optional[ActivationTensors]
    # other run state configuration
    var batch_size: Int  # the batch size (B) of current forward pass
    var seq_len: Int  # the sequence length (T) of current forward pass
    var inputs: Optional[IntPtr]  # the input tokens for the current forward pass
    var targets: Optional[IntPtr]  # the target tokens for the current forward pass
    var mean_loss: FLOAT  # after a forward pass with targets, will be populated with the mean loss
    var checkpoint_path: String

    def __init__(out self, checkpoint_path: String) raises:
        self.checkpoint_path = checkpoint_path

        self.param_sizes = InlineArray[Int, NUM_PARAMETER_TENSORS](fill=0)
        self.act_sizes = InlineArray[Int, NUM_ACTIVATION_TENSORS](fill=0)

        model_file = open(checkpoint_path, "r")
        model_header = InlineArray[Int32, 256](fill=0)

        read_to_buf(model_header, model_file)

        if model_header[0] != 20240326:
            raise Error(t"Bad magic model file {model_header[0]}")
        if model_header[1] != 3:
            raise Error(t"Bad version in model file {model_header[1]}")

        # read in hyperparameters

        self.config = GPT2Config(
            Int(model_header[2]),
            Int(model_header[3]),
            Int(model_header[4]),
            Int(model_header[5]),
            Int(model_header[6]),
            Int(model_header[7]),
        )

        var maxT: Int = self.config.max_seq_len
        var L: Int = self.config.num_layers
        var C: Int = self.config.channels
        var Vp: Int = self.config.padded_vocab_size

        # allocate space for all the parameters and read them in
        self.param_sizes[0] = Vp * C
        self.param_sizes[1] = maxT * C
        self.param_sizes[2] = L * C
        self.param_sizes[3] = L * C
        self.param_sizes[4] = L * (3 * C) * C
        self.param_sizes[5] = L * (3 * C)
        self.param_sizes[6] = L * C * C
        self.param_sizes[7] = L * C
        self.param_sizes[8] = L * C
        self.param_sizes[9] = L * C
        self.param_sizes[10] = L * (4 * C) * C
        self.param_sizes[11] = L * (4 * C)
        self.param_sizes[12] = L * C * (4 * C)
        self.param_sizes[13] = L * C
        self.param_sizes[14] = C
        self.param_sizes[15] = C

        # cound the number of paramaters
        var num_parameters: Int = 0

        for i in range(NUM_PARAMETER_TENSORS):
            num_parameters += self.param_sizes[i]

        self.num_parameters = num_parameters

        # read in all the parameters from file
        self.params = ParameterTensors(self.param_sizes)
        self.params_memory = self.params.params_memory

        read_to_dtype_pointer[DType.float32](
            self.params_memory, model_file, num_parameters
        )
        model_file.close()

        # other inits (lazily allocated later)
        self.acts = None
        self.num_activations = 0  # for now

        self.grads = None
        self.grads_acts = None
        self.m_memory = None
        self.v_memory = None
        self.inputs = None
        self.targets = None
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0  # -1.0 will designate no loss

        print("[GPT-2]")
        print("max_seq_len:", self.config.max_seq_len)
        print("vocab_size:", self.config.vocab_size)
        print("padded_vocab_size:", self.config.padded_vocab_size)
        print("num_layers:", self.config.num_layers)
        print("num_heads:", self.config.num_heads)
        print("channels:", self.config.channels)

        print("num_parameters:", num_parameters)


    def forward(
        mut self,
        inputs: IntPtr,
        targets: Optional[IntPtr],
        B: Int,
        T: Int,
    ):
        # targets are optional and could be None

        # convenience parameters
        var V: Int = self.config.vocab_size
        var Vp: Int = self.config.padded_vocab_size
        var L: Int = self.config.num_layers
        var NH: Int = self.config.num_heads
        var C: Int = self.config.channels

        # allocate space for all the activations if needed (done here, lazily)
        if not self.acts:
            # record the current B,T as well
            self.batch_size = B
            self.seq_len = T

            # and now allocate the space
            self.act_sizes[0] = B * T * C
            self.act_sizes[1] = L * B * T * C
            self.act_sizes[2] = L * B * T
            self.act_sizes[3] = L * B * T
            self.act_sizes[4] = L * B * T * 3 * C
            self.act_sizes[5] = L * B * T * C
            self.act_sizes[6] = L * B * NH * T * T
            self.act_sizes[7] = L * B * NH * T * T
            self.act_sizes[8] = L * B * T * C
            self.act_sizes[9] = L * B * T * C
            self.act_sizes[10] = L * B * T * C
            self.act_sizes[11] = L * B * T
            self.act_sizes[12] = L * B * T
            self.act_sizes[13] = L * B * T * 4 * C
            self.act_sizes[14] = L * B * T * 4 * C
            self.act_sizes[15] = L * B * T * C
            self.act_sizes[16] = L * B * T * C
            self.act_sizes[17] = B * T * C
            self.act_sizes[18] = B * T
            self.act_sizes[19] = B * T
            self.act_sizes[20] = B * T * Vp
            self.act_sizes[21] = B * T * Vp
            self.act_sizes[22] = B * T

            var num_activations: Int = 0
            for i in range(NUM_ACTIVATION_TENSORS):
                num_activations += self.act_sizes[i]

            print("num_activations:", num_activations)

            self.acts = ActivationTensors(self.act_sizes)
            self.num_activations = num_activations
            # also create memory for caching inputs and targets

            self.inputs = alloc[Scalar[dtype_int]](B * T)
            self.targets = alloc[Scalar[dtype_int]](B * T)

        else:
            # validate B,T is no larger than what was previously allocated
            # in principle, we could re-allocate a larger chunk of memory, for now we just error out
            if B > self.batch_size or T > self.seq_len:
                print("Error: batch size or sequence length is inadequately large")
                # print("Model: B=%d T=%d, Desired: B=%d T=%d\n", self.batch_size, self.seq_len, B, T)

        # cache the inputs/targets
        memcpy(dest=self.inputs.value(), src=inputs, count=B * T)

        if targets:
            memcpy(dest=self.targets.value(), src=targets.value(), count=B * T)

        # forward pass

        ref acts = self.acts.value()
        var residual: FloatPtr
        encoder_forward(
            acts.encoded, inputs, self.params.wte, self.params.wpe, B, T, C
        )  # encoding goes into residual[0]

        for l in range(L):
            residual = acts.residual3 + (l - 1) * B * T * C

            if l == 0:
                residual = acts.encoded

            # get the pointers of the weights for this layer
            var l_ln1w: FloatPtr = self.params.ln1w + l * C
            var l_ln1b: FloatPtr = self.params.ln1b + l * C
            var l_qkvw: FloatPtr = (
                self.params.qkvw + l * 3 * C * C
            )
            var l_qkvb: FloatPtr = self.params.qkvb + l * 3 * C
            var l_attprojw: FloatPtr = (
                self.params.attprojw + l * C * C
            )
            var l_attprojb: FloatPtr = (
                self.params.attprojb + l * C
            )
            var l_ln2w: FloatPtr = self.params.ln2w + l * C
            var l_ln2b: FloatPtr = self.params.ln2b + l * C
            var l_fcw: FloatPtr = (
                self.params.fcw + l * 4 * C * C
            )
            var l_fcb: FloatPtr = self.params.fcb + l * 4 * C
            var l_fcprojw: FloatPtr = (
                self.params.fcprojw + l * C * 4 * C
            )
            var l_fcprojb: FloatPtr = (
                self.params.fcprojb + l * C
            )

            # get the pointers of the activations for this layer
            var l_ln1: FloatPtr = acts.ln1 + l * B * T * C
            var l_ln1_mean: FloatPtr = (
                acts.ln1_mean + l * B * T
            )
            var l_ln1_rstd: FloatPtr = (
                acts.ln1_rstd + l * B * T
            )
            var l_qkv: FloatPtr = (
                acts.qkv + l * B * T * 3 * C
            )
            var l_atty: FloatPtr = (
                acts.atty + l * B * T * C
            )
            var l_preatt: FloatPtr = (
                acts.preatt + l * B * NH * T * T
            )
            var l_att: FloatPtr = (
                acts.att + l * B * NH * T * T
            )
            var l_attproj: FloatPtr = (
                acts.attproj + l * B * T * C
            )
            var l_residual2: FloatPtr = (
                acts.residual2 + l * B * T * C
            )
            var l_ln2: FloatPtr = acts.ln2 + l * B * T * C
            var l_ln2_mean: FloatPtr = (
                acts.ln2_mean + l * B * T
            )
            var l_ln2_rstd: FloatPtr = (
                acts.ln2_rstd + l * B * T
            )
            var l_fch: FloatPtr = (
                acts.fch + l * B * T * 4 * C
            )
            var l_fch_gelu: FloatPtr = (
                acts.fch_gelu + l * B * T * 4 * C
            )
            var l_fcproj: FloatPtr = (
                acts.fcproj + l * B * T * C
            )
            var l_residual3: FloatPtr = (
                acts.residual3 + l * B * T * C
            )

            # now do the forward pass

            layernorm_forward(
                l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C
            )
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C)
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH)
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
            residual_forward(l_residual2, residual, l_attproj, B * T * C)
            layernorm_forward(
                l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C
            )
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C)
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C)
            matmul_forward(
                l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C
            )
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C)

        residual = (
            acts.residual3 + (L - 1) * B * T * C
        )  # last residual is in residual3
        layernorm_forward(
            acts.lnf,
            acts.lnf_mean,
            acts.lnf_rstd,
            residual,
            self.params.lnfw,
            self.params.lnfb,
            B,
            T,
            C,
        )
        matmul_forward(
            acts.logits, acts.lnf, self.params.wte, None, B, T, C, Vp
        )
        softmax_forward(acts.probs, acts.logits, B, T, V, Vp)

        # also forward the cross-entropy loss function if we have the targets
        if targets:
            crossentropy_forward(
                acts.losses, acts.probs, targets.value(), B, T, Vp
            )
            # for convenience also evaluate the mean loss
            var mean_loss: FLOAT = 0.0
            for i in range(B * T):
                mean_loss += acts.losses[i]
            mean_loss /= FLOAT(B * T)
            self.mean_loss = mean_loss
        else:
            # if we don't have targets, we don't have a loss
            self.mean_loss = -1.0


    def zero_grad(mut self):
        if self.grads:
            memset_zero(self.grads.value().params_memory, self.num_parameters)

        if self.grads_acts:
            memset_zero(self.grads_acts.value().acts_memory, self.num_activations)


    def backward(mut self):
        # double check we forwarded previously, with targets
        if self.mean_loss == -1.0:
            print("Error: must forward with targets before backward\n")

        # lazily allocate the memory for gradients of the weights and activations, if needed
        if not self.grads:
            self.grads = ParameterTensors(self.param_sizes)
            self.grads_acts = ActivationTensors(self.act_sizes)
            self.zero_grad()

        # convenience shortcuts
        var B: Int = self.batch_size
        var T: Int = self.seq_len
        var V: Int = self.config.vocab_size
        var Vp: Int = self.config.padded_vocab_size
        var L: Int = self.config.num_layers
        var NH: Int = self.config.num_heads
        var C: Int = self.config.channels

        # references to the (now guaranteed allocated) tensor structs
        ref acts = self.acts.value()
        ref grads = self.grads.value()
        ref grads_acts = self.grads_acts.value()

        # backward pass

        # we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        # technically this is a small, inline backward() pass of calculating
        # total, final loss as the mean over all losses over all (B,T) positions in the batch

        var dloss_mean: FLOAT = FLOAT(1.0) / FLOAT(B * T)

        # Fill the gradient losses with the mean loss value
        def _fill_losses[width: Int](i: Int) {grads_acts, dloss_mean}:
            grads_acts.losses.store[width=width](i, dloss_mean)

        vectorize[SIMD_WIDTH](B * T, _fill_losses)

        crossentropy_softmax_backward(
            grads_acts.logits,
            grads_acts.losses,
            acts.probs,
            self.targets.value(),
            B,
            T,
            V,
            Vp,
        )
        matmul_backward(
            grads_acts.lnf,
            grads.wte,
            None,
            grads_acts.logits,
            acts.lnf,
            self.params.wte,
            B,
            T,
            C,
            Vp,
        )
        var residual: FloatPtr = (
            acts.residual3 + (L - 1) * B * T * C
        )  # last layer's residual
        var dresidual: FloatPtr = (
            grads_acts.residual3 + (L - 1) * B * T * C
        )  # write to last layer's residual
        layernorm_backward(
            dresidual,
            grads.lnfw,
            grads.lnfb,
            grads_acts.lnf,
            residual,
            self.params.lnfw,
            acts.lnf_mean,
            acts.lnf_rstd,
            B,
            T,
            C,
        )

        for l in range(L - 1, -1, -1):
            var residual = acts.encoded
            var dresidual = grads_acts.encoded
            if l != 0:
                residual = acts.residual3 + (l - 1) * B * T * C
                dresidual = grads_acts.residual3 + (l - 1) * B * T * C

            # get the pointers of the weights for this layer
            var l_ln1w: FloatPtr = self.params.ln1w + l * C
            var l_qkvw: FloatPtr = (
                self.params.qkvw + l * 3 * C * C
            )
            var l_attprojw: FloatPtr = (
                self.params.attprojw + l * C * C
            )
            var l_ln2w: FloatPtr = self.params.ln2w + l * C
            var l_fcw: FloatPtr = (
                self.params.fcw + l * 4 * C * C
            )
            var l_fcprojw: FloatPtr = (
                self.params.fcprojw + l * C * 4 * C
            )
            # get the pointers of the gradients of the weights for this layer
            var dl_ln1w: FloatPtr = grads.ln1w + l * C
            var dl_ln1b: FloatPtr = grads.ln1b + l * C
            var dl_qkvw: FloatPtr = (
                grads.qkvw + l * 3 * C * C
            )
            var dl_qkvb: FloatPtr = grads.qkvb + l * 3 * C
            var dl_attprojw: FloatPtr = (
                grads.attprojw + l * C * C
            )
            var dl_attprojb: FloatPtr = (
                grads.attprojb + l * C
            )
            var dl_ln2w: FloatPtr = grads.ln2w + l * C
            var dl_ln2b: FloatPtr = grads.ln2b + l * C
            var dl_fcw: FloatPtr = (
                grads.fcw + l * 4 * C * C
            )
            var dl_fcb: FloatPtr = grads.fcb + l * 4 * C
            var dl_fcprojw: FloatPtr = (
                grads.fcprojw + l * C * 4 * C
            )
            var dl_fcprojb: FloatPtr = (
                grads.fcprojb + l * C
            )
            # get the pointers of the activations for this layer
            var l_ln1: FloatPtr = acts.ln1 + l * B * T * C
            var l_ln1_mean: FloatPtr = (
                acts.ln1_mean + l * B * T
            )
            var l_ln1_rstd: FloatPtr = (
                acts.ln1_rstd + l * B * T
            )
            var l_qkv: FloatPtr = (
                acts.qkv + l * B * T * 3 * C
            )
            var l_atty: FloatPtr = (
                acts.atty + l * B * T * C
            )
            var l_att: FloatPtr = (
                acts.att + l * B * NH * T * T
            )
            var l_residual2: FloatPtr = (
                acts.residual2 + l * B * T * C
            )
            var l_ln2: FloatPtr = acts.ln2 + l * B * T * C
            var l_ln2_mean: FloatPtr = (
                acts.ln2_mean + l * B * T
            )
            var l_ln2_rstd: FloatPtr = (
                acts.ln2_rstd + l * B * T
            )
            var l_fch: FloatPtr = (
                acts.fch + l * B * T * 4 * C
            )
            var l_fch_gelu: FloatPtr = (
                acts.fch_gelu + l * B * T * 4 * C
            )
            # get the pointers of the gradients of the activations for this layer
            var dl_ln1: FloatPtr = (
                grads_acts.ln1 + l * B * T * C
            )
            var dl_qkv: FloatPtr = (
                grads_acts.qkv + l * B * T * 3 * C
            )
            var dl_atty: FloatPtr = (
                grads_acts.atty + l * B * T * C
            )
            var dl_preatt: FloatPtr = (
                grads_acts.preatt + l * B * NH * T * T
            )
            var dl_att: FloatPtr = (
                grads_acts.att + l * B * NH * T * T
            )
            var dl_attproj: FloatPtr = (
                grads_acts.attproj + l * B * T * C
            )
            var dl_residual2: FloatPtr = (
                grads_acts.residual2 + l * B * T * C
            )
            var dl_ln2: FloatPtr = (
                grads_acts.ln2 + l * B * T * C
            )
            var dl_fch: FloatPtr = (
                grads_acts.fch + l * B * T * 4 * C
            )
            var dl_fch_gelu: FloatPtr = (
                grads_acts.fch_gelu + l * B * T * 4 * C
            )
            var dl_fcproj: FloatPtr = (
                grads_acts.fcproj + l * B * T * C
            )
            var dl_residual3: FloatPtr = (
                grads_acts.residual3 + l * B * T * C
            )

            # backprop this layer
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C)
            matmul_backward(
                dl_fch_gelu,
                dl_fcprojw,
                dl_fcprojb,
                dl_fcproj,
                l_fch_gelu,
                l_fcprojw,
                B,
                T,
                4 * C,
                C,
            )
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C)
            matmul_backward(
                dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C
            )
            layernorm_backward(
                dl_residual2,
                dl_ln2w,
                dl_ln2b,
                dl_ln2,
                l_residual2,
                l_ln2w,
                l_ln2_mean,
                l_ln2_rstd,
                B,
                T,
                C,
            )
            residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C)
            matmul_backward(
                dl_atty,
                dl_attprojw,
                dl_attprojb,
                dl_attproj,
                l_atty,
                l_attprojw,
                B,
                T,
                C,
                C,
            )
            attention_backward(
                dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH
            )
            matmul_backward(
                dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C
            )
            layernorm_backward(
                dresidual,
                dl_ln1w,
                dl_ln1b,
                dl_ln1,
                residual,
                l_ln1w,
                l_ln1_mean,
                l_ln1_rstd,
                B,
                T,
                C,
            )

        encoder_backward(
            grads.wte,
            grads.wpe,
            grads_acts.encoded,
            self.inputs.value(),
            B,
            T,
            C,
        )


    def update(
        mut self,
        learning_rate: FLOAT,
        beta1: FLOAT,
        beta2: FLOAT,
        eps: FLOAT,
        weight_decay: FLOAT,
        t: Int,
    ):
        # reference: https:#pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        # lazily allocate the memory for m_memory and v_memory
        if not self.m_memory:
            self.m_memory = alloc[Scalar[dtype]](self.num_parameters)
            self.v_memory = alloc[Scalar[dtype]](self.num_parameters)

            memset_zero(self.m_memory.value(), self.num_parameters)
            memset_zero(self.v_memory.value(), self.num_parameters)

        # local copies of the pointers for the update loop
        var params_memory = self.params_memory
        var grads_memory = self.grads.value().params_memory
        var m_memory = self.m_memory.value()
        var v_memory = self.v_memory.value()

        var num_vectorize = self.num_parameters // NUM_PARALLELIZE

        @parameter
        def _calc(ip: Int):
            def _op[width: Int](_iv: Int) {ip, num_vectorize, params_memory, grads_memory, m_memory, v_memory, learning_rate, beta1, beta2, eps, weight_decay, t}:
                var iv = ip * num_vectorize + _iv
                var param = params_memory.load[width=width](iv)
                var grad = grads_memory.load[width=width](iv)

                # update the first moment (momentum)
                var m = (
                    beta1 * m_memory.load[width=width](iv)
                    + (1.0 - beta1) * grad
                )
                # update the second moment (RMSprop)
                var v = (
                    beta2 * v_memory.load[width=width](iv)
                    + (1.0 - beta2) * grad * grad
                )
                # bias-correct both moments
                var m_hat = m / (1.0 - pow(beta1, t))
                var v_hat = v / (1.0 - pow(beta2, t))

                # update
                m_memory.store(iv, m)
                v_memory.store(iv, v)
                params_memory.store(
                    iv,
                    params_memory.load[width=width](iv)
                    - learning_rate
                    * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param),
                )

            vectorize[SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize, _op)

        parallelize[_calc](NUM_PARALLELIZE)


    def __del__(deinit self):
        self.params_memory.free()
        if self.grads:
            self.grads.value().params_memory.free()
        if self.m_memory:
            self.m_memory.value().free()
        if self.v_memory:
            self.v_memory.value().free()
        if self.acts:
            self.acts.value().acts_memory.free()
        if self.grads_acts:
            self.grads_acts.value().acts_memory.free()
        if self.inputs:
            self.inputs.value().free()
        if self.targets:
            self.targets.value().free()


# ifndef TESTING
# if we are TESTING (see test_gpt2.c), we'll skip the maiN:Int32 below

# ----------------------------------------------------------------------------
# data loader lite
# returns random batches of data from a file of integers


struct DataLoader[B: Int, T: Int]:
    # hyperparameters

    # input handling and its state
    var filename: String
    var tokens_file: FileHandle
    var file_size: Int
    var current_position: Int
    # output memory
    var batch: InlineArray[Scalar[dtype_int], Self.B * Self.T + 1]
    var inputs: IntPtr
    var targets: IntPtr
    # convenience variables
    var num_batches: Int

    def __init__(out self, filename: String) raises:
        self.filename = filename
        self.tokens_file = open(filename, "r")

        # determine the file size
        self.file_size = getsize(filename)

        if self.file_size < (Self.B * Self.T + 1) * SIZEOF_INT:
            raise Error(
                "Error: file size is too small for the batch size and"
                " sequence length"
            )

        self.current_position = 0  # start at the beginning

        # allocate space for B*T + 1 integers to store the inputs and targets
        self.batch = InlineArray[Scalar[dtype_int], Self.B * Self.T + 1](fill=0)
        self.inputs = self.batch.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.targets = (
            self.inputs + 1
        )  # targets are shifted by one
        self.num_batches = self.file_size // (Self.B * Self.T * SIZEOF_INT)

    def reset(mut self):
        self.current_position = 0

    def next_batch(mut self) raises:
        # if we are at the end of the file, loop back to the beginning
        if (
            self.current_position + ((Self.B * Self.T + 1) * SIZEOF_INT)
            > self.file_size
        ):
            self.current_position = 0

        # read the B*T+1 integers from the file into batch
        _ = self.tokens_file.seek(UInt64(self.current_position))

        read_to_buf(self.batch, self.tokens_file)

        # advance the current position by B*T integers
        self.current_position += Self.B * Self.T * SIZEOF_INT

    # no explicit free needed: the FileHandle closes itself on destruction
    # and the batch InlineArray lives inline in the struct


# ----------------------------------------------------------------------------
# sampler


def random_u32(mut state: UInt64) -> UInt32:
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    return ((state * RU32_HEX) >> 32).cast[DType.uint32]()


def random_f32(mut state: UInt64) -> Float32:
    return (random_u32(state) >> 8).cast[DType.float32]() / RF32_DIV


def sample_mult(
    probabilities: FloatPtr, n: Int, coin: FLOAT
) -> Int:
    # sample index from probabilities (they must sum to 1!)
    # coin is a random number in [0, 1), usually from random_f32()
    var cdf: FLOAT = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if coin < cdf:
            return i
    return n - 1


# ----------------------------------------------------------------------------
# Tokenizer (only supports decoding)
# this mojo version needs refinements, still buggy


struct Tokenizer:
    var vocab_size: Int
    var token_table: List[String]
    var init_ok: Int

    def __init__(out self, filename: String) raises:
        self.vocab_size = 0
        self.token_table = List[String]()
        self.init_ok = 0

        var file: FileHandle

        try:
            file = open(filename, "r")
        except:
            print("---")
            print("WARNING: Failed to open the tokenizer file", filename)
            print("The Tokenizer is a new feature added April 14 2024.")
            print("Re-run `python train_gpt2.py` to write it")
            print("---")

            self.init_ok = 0
            return

        var header = InlineArray[Int32, 256](fill=0)
        _ = file.read(header)

        if header[0] != 20240328:
            raise Error(t"Bad magic tokenizer file {header[0]}")
        if header[1] != 2:
            raise Error(t"Bad version in tokenizer file {header[1]}")

        self.vocab_size = Int(header[2])

        for _ in range(self.vocab_size):
            var length = Int(file.read_bytes(1)[0])
            var str = String(from_utf8_lossy=file.read_bytes(length))
            if length > 0 and str.byte_length() > 0:
                self.token_table.append(str)
            else:
                self.token_table.append("")

        file.close()
        self.init_ok = 1

    def decode(self, token_id: Int) -> String:
        if self.init_ok == 0:
            return ""

        if token_id >= 0 and token_id < self.vocab_size:
            return self.token_table[token_id]
        else:
            return ""

    def safe_printf(self, s: String):
        # the tokens are raw bytes, and we we only want to print the printable ones
        # many bytes can be various control codes, backspace, etc.
        if s.byte_length() == 0:
            return
        if s.as_bytes()[0] == 0:
            return
        # handle individual byte tokens
        # every token is asserted to be at least one byte so doing piece[1] is ok

        ### --- TODO
        # if (s[1] == '\0') {
        # unsigned char byte_val = piece[0];
        # if (!(isprint(byte_val) || isspace(byte_val))) {
        #    return; // weird byte, don't print it
        # }
        # }

        print(s, end="")


def read_to_buf[
    T: DType, n: Int
](mut buf: InlineArray[Scalar[T], n], file_handle: FileHandle) raises -> None:
    _ = file_handle.read(buf)


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


# ----------------------------------------------------------------------------
# main training loop


def main() raises:
    # build the GPT-2 model from a checkpoint
    var model = GPT2("gpt2_124M.bin")

    # build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    var tiny_stories_train: String = "./data/TinyStories_train.bin"
    var tiny_stories_val: String = "./data/TinyStories_val.bin"
    var tiny_shakespeare_train: String = "./data/tiny_shakespeare_train.bin"
    var tiny_shakespeare_val: String = "./data/tiny_shakespeare_val.bin"

    var train_tokens: String = tiny_shakespeare_train
    var val_tokens: String = tiny_shakespeare_val

    try:
        var file = open(tiny_shakespeare_train, "r")
        file.close()
    except:
        # both in one go ...
        train_tokens = tiny_stories_train
        val_tokens = tiny_stories_val

    comptime B: Int = 4
    comptime T: Int = 64
    var train_loader = DataLoader[B, T](train_tokens)
    print("train dataset num_batches:", train_loader.num_batches)
    var val_loader = DataLoader[B, T](val_tokens)
    print("val dataset num_batches:", val_loader.num_batches)
    var val_num_batches: Int = 10

    # build the Tokenizer
    var tokenizer = Tokenizer("gpt2_tokenizer.bin")

    # some memory for generating samples from the model
    var rng_state: UInt64 = 1337
    var gen_max_length: Int = 64
    var gen_tokens = alloc[Scalar[dtype_int]](gen_max_length)

    # train

    var elapsed_time_ms_total = 0.0

    for step in range(41):
        # once in a while estimate the validation loss
        if step % 10 == 0:
            var val_loss: FLOAT = 0.0
            val_loader.reset()
            for _ in range(val_num_batches):
                val_loader.next_batch()
                model.forward(val_loader.inputs, val_loader.targets, B, T)
                val_loss += model.mean_loss

            val_loss /= FLOAT(val_num_batches)
            print("val loss", val_loss)

        # once in a while do model inference to prgenerated INT32 text
        if step > 0 and step % 20 == 0:
            gen_tokens[
                0
            ] = GPT2_EOT  # the GPT-2 EOT token kicks off the generation

            print("generating:\n---")
            for t in range(1, gen_max_length):
                # note that inference is wasteful here because
                # for each t, we re-compute all activations between 0 and t
                # leaving this alone because you want separate code for inference anyway
                # the inference here is just for sanity checking purposes
                model.forward(gen_tokens, None, 1, t)
                var probs = (
                    model.acts.value().probs
                    + (t - 1) * model.config.padded_vocab_size
                )
                var coin: FLOAT = random_f32(rng_state).cast[dtype]()
                var next_token: Int = sample_mult(
                    probs, model.config.vocab_size, coin
                )
                gen_tokens[t] = INT(next_token)
                # print the generated token, either using the Tokenizer or a fallback
                if tokenizer.init_ok:
                    var token_str: String = tokenizer.decode(next_token)
                    tokenizer.safe_printf(token_str)

                else:
                    # fall back to printing the token id
                    print(next_token, end=" ")

            print("\n---")

        # do a training step

        var start_time = perf_counter_ns()

        train_loader.next_batch()
        model.forward(train_loader.inputs, train_loader.targets, B, T)
        model.zero_grad()
        model.backward()
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1)

        var elapsed_time_ms = Float64(perf_counter_ns() - start_time) / 1_000_000.0

        elapsed_time_ms_total += elapsed_time_ms

        var tokens_per_second = Float64(B * T) * 1000.0 / elapsed_time_ms

        print(
            t"step {step}: train loss {model.mean_loss} (took {Int(elapsed_time_ms)} ms, average: {Int(elapsed_time_ms_total / Float64(step + 1))} ms, {Int(tokens_per_second)} tok/s)"
        )
