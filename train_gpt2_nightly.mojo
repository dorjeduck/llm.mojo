from algorithm import vectorize, parallelize
from collections.vector import InlinedFixedVector
from math import sqrt, exp, tanh, cosh, log
from memory import UnsafePointer,memset, memset_zero, memcpy
from python import Python
from sys import exit
from sys.info import is_apple_silicon,simdwidthof,sizeof
from time import now
from utils import StringRef

fn get_simd_width() -> Int:
    if is_apple_silicon():
        return 4 * simdwidthof[dtype]()
    else:
        return 2 * simdwidthof[dtype]()

alias dtype = DType.float32 # must be float32 for now
alias dtype_int = DType.int32 # must be int32 for now

alias SIZEOF_FLOAT = sizeof[dtype]()
alias SIZEOF_INT = sizeof[dtype_int]()

alias SIMD_WIDTH = get_simd_width()

alias RU32_HEX = 0x2545F4914F6CDD1D
alias RF32_DIV = 16777216.0



alias FLOAT = SIMD[dtype, 1]
alias INT = SIMD[dtype_int, 1]

alias NULL = UnsafePointer[Scalar[dtype]]()
alias NULL_INT = UnsafePointer[Scalar[dtype_int]]()
alias M_PI: FLOAT = 3.141592653589793115997963468544185161590576171875

alias GPT2_EOT = 50256

alias NUM_PARALLELIZE = 8
alias UNROLL_FACTOR = 4


## ----------------------------------------------------------------------------
# all the individual layers' forward and backward passes


fn encoder_forward(
    out: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype_int]],
    wte: UnsafePointer[Scalar[dtype]],
    wpe: UnsafePointer[Scalar[dtype]],
    B: Int,
    T: Int,
    C: Int,
):
    @parameter
    fn _calc(b: Int):
        for t in range(T):
            # seek to the output position in out[b,t,:]
            var out_bt: UnsafePointer[Scalar[dtype]] = out + b * T * C + t * C
            # get the index of the token at inp[b, t]
            var ix = inp[b * T + t]
            # seek to the position in wte corresponding to the token
            var wte_ix: UnsafePointer[Scalar[dtype]] = wte + int(ix) * C
            # seek to the position in wpe corresponding to the position
            var wpe_t: UnsafePointer[Scalar[dtype]] = wpe + t * C
            # add the two vectors and store the result in out[b,t,:]

            @parameter
            fn _op[width: Int](iv: Int):
                out_bt.store[width=width](
                    iv, wte_ix.load[width=width](iv) + wpe_t.load[width=width](iv)
                )

            vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

    parallelize[_calc](B)


fn encoder_backward(
    dwte: UnsafePointer[Scalar[dtype]],
    dwpe: UnsafePointer[Scalar[dtype]],
    dout: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype_int]],
    B: Int,
    T: Int,
    C: Int,
):
    @parameter
    fn _calc(b: Int):
        for t in range(T):
            var dout_bt: UnsafePointer[Scalar[dtype]] = dout + b * T * C + t * C
            var ix = inp[b * T + t]
            var dwte_ix: UnsafePointer[Scalar[dtype]] = dwte + int(ix) * C
            var dwpe_t: UnsafePointer[Scalar[dtype]] = dwpe + t * C

            @parameter
            fn _op[width: Int](iv: Int):
                var d = dout_bt.load[width=width](iv)
                dwte_ix.store[width=width](iv, dwte_ix.load[width=width](iv) + d)
                dwpe_t.store[width=width](iv, dwpe_t.load[width=width](iv) + d)

            vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

    parallelize[_calc](B)


fn layernorm_forward(
    out: UnsafePointer[Scalar[dtype]],
    mean: UnsafePointer[Scalar[dtype]],
    rstd: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    weight: UnsafePointer[Scalar[dtype]],
    bias: UnsafePointer[Scalar[dtype]],
    B: Int,
    T: Int,
    C: Int,
):
    var eps: FLOAT = 1e-5

    @parameter
    fn _calc(b: Int):
        for t in range(T):
            # seek to the input position inp[b,t,:]
            var x: UnsafePointer[Scalar[dtype]] = inp + b * T * C + t * C
            # calculate the mean
            var m: FLOAT = 0.0

            @parameter
            fn _op[width: Int](iv: Int):
                m += x.load[width=width](iv).reduce_add[1]()

            vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

            m = m / C

            # calculate the variance (without any bias correction)
            var v: FLOAT = 0.0

            @parameter
            fn _op2[width: Int](iv: Int):
                var xshift = x.load[width=width](iv) - m
                v += pow(xshift, 2).reduce_add[1]()

            vectorize[_op2, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

            v = v / C

            # calculate the rstd
            var s: FLOAT = FLOAT(1.0) / sqrt(v + eps)

            # seek to the output position in out[b,t,:]
            var out_bt: UnsafePointer[Scalar[dtype]] = out + b * T * C + t * C

            @parameter
            fn _op3[width: Int](iv: Int):
                var n = s * (x.load[width=width](iv) - m)  # normalized output
                out_bt.store[width=width](
                    iv, n * weight.load[width=width](iv) + bias.load[width=width](iv)
                )  # scale and shift it

            vectorize[_op3, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

            # cache the mean and rstd for the backward pass later
            mean[b * T + t] = m
            rstd[b * T + t] = s

    parallelize[_calc](B)


fn layernorm_backward(
    dinp: UnsafePointer[Scalar[dtype]],
    dweight: UnsafePointer[Scalar[dtype]],
    dbias: UnsafePointer[Scalar[dtype]],
    dout: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    weight: UnsafePointer[Scalar[dtype]],
    mean: UnsafePointer[Scalar[dtype]],
    rstd: UnsafePointer[Scalar[dtype]],
    B: Int,
    T: Int,
    C: Int,
):
    @parameter
    fn _calc(b: Int):
        for t in range(T):
            var dout_bt: UnsafePointer[Scalar[dtype]] = dout + b * T * C + t * C
            var inp_bt: UnsafePointer[Scalar[dtype]] = inp + b * T * C + t * C
            var dinp_bt: UnsafePointer[Scalar[dtype]] = dinp + b * T * C + t * C
            var mean_bt: FLOAT = mean[b * T + t]
            var rstd_bt: FLOAT = rstd[b * T + t]

            # first: two reduce operations
            var dnorm_mean: FLOAT = 0.0
            var dnorm_norm_mean: FLOAT = 0.0

            @parameter
            fn _op[width: Int](iv: Int):
                var norm_bti = (inp_bt.load[width=width](iv) - mean_bt) * rstd_bt
                var dnorm_i = weight.load[width=width](iv) * dout_bt.load[width=width](
                    iv
                )
                dnorm_mean += dnorm_i.reduce_add[1]()
                dnorm_norm_mean += (dnorm_i * norm_bti).reduce_add[1]()

            vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

            dnorm_mean = dnorm_mean / C
            dnorm_norm_mean = dnorm_norm_mean / C

            # now iterate again and accumulate all the gradients

            @parameter
            fn _op2[width: Int](iv: Int):
                var norm_bti = (inp_bt.load[width=width](iv) - mean_bt) * rstd_bt
                var dnorm_i = weight.load[width=width](iv) * dout_bt.load[width=width](
                    iv
                )
                # gradient contribution to bias
                dbias.store[width=width](
                    iv, dbias.load[width=width](iv) + dout_bt.load[width=width](iv)
                )
                # gradient contribution to weight
                dweight.store[width=width](
                    iv,
                    dweight.load[width=width](iv)
                    + norm_bti * dout_bt.load[width=width](iv),
                )
                # gradient contribution to input
                dinp_bt.store[width=width](
                    iv,
                    dinp_bt.load[width=width](iv)
                    + (dnorm_i - dnorm_mean - (norm_bti * dnorm_norm_mean)) * rstd_bt,
                )

            vectorize[_op2, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

    parallelize[_calc](B)


fn matmul_forward(
    out: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    weight: UnsafePointer[Scalar[dtype]],
    bias: UnsafePointer[Scalar[dtype]],
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
    fn _calc(b: Int):
        for t in range(T):
            var out_bt: UnsafePointer[Scalar[dtype]] = out + b * T * OC + t * OC
            var inp_bt: UnsafePointer[Scalar[dtype]] = inp + b * T * C + t * C

            for o in range(OC):
                var val: FLOAT = 0.0
                if bias != NULL:
                    val = bias[o]
                var wrow: UnsafePointer[Scalar[dtype]] = weight + o * C

                @parameter
                fn _op[width: Int](iv: Int):
                    var t = inp_bt.load[width=width](iv) * wrow.load[width=width](iv)
                    val += t.reduce_add[1]()

                vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

                out_bt[o] = val

    parallelize[_calc](B)


fn matmul_backward(
    dinp: UnsafePointer[Scalar[dtype]],
    dweight: UnsafePointer[Scalar[dtype]],
    dbias: UnsafePointer[Scalar[dtype]],
    dout: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    weight: UnsafePointer[Scalar[dtype]],
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
    fn _calc(b: Int):
        for t in range(T):
            var dout_bt: UnsafePointer[Scalar[dtype]] = dout + b * T * OC + t * OC
            var dinp_bt: UnsafePointer[Scalar[dtype]] = dinp + b * T * C + t * C
            for o in range(OC):
                var wrow: UnsafePointer[Scalar[dtype]] = weight + o * C
                var d: FLOAT = dout_bt[o]

                @parameter
                fn _op[width: Int](iv: Int):
                    dinp_bt.store[width=width](
                        iv,
                        dinp_bt.load[width=width](iv) + wrow.load[width=width](iv) * d,
                    )  # scale and shift it

                vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

    parallelize[_calc](B)
    # backward into weight/bias, parallelize over output channels OC
    # pragma omp parallel for

    @parameter
    fn _calc2(o: Int):
        for b in range(B):
            for t in range(T):
                var dout_bt: UnsafePointer[Scalar[dtype]] = dout + b * T * OC + t * OC
                var inp_bt: UnsafePointer[Scalar[dtype]] = inp + b * T * C + t * C
                var dwrow: UnsafePointer[Scalar[dtype]] = dweight + o * C
                var d: FLOAT = dout_bt[o]
                if dbias != NULL:
                    dbias[o] += d

                @parameter
                fn _op[width: Int](iv: Int):
                    dwrow.store[width=width](
                        iv,
                        dwrow.load[width=width](iv) + inp_bt.load[width=width](iv) * d,
                    )  # scale and shift it

                vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=C)

    parallelize[_calc2](OC)


fn attention_forward(
    out: UnsafePointer[Scalar[dtype]],
    preatt: UnsafePointer[Scalar[dtype]],
    att: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
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
    var scale: FLOAT = FLOAT(1.0) / sqrt(hs)

    # pragma omp parallel for collapse(3)
    @parameter
    fn _calc(b: Int):
        # for b in range(B):
        for t in range(T):
            for h in range(NH):
                var query_t: UnsafePointer[Scalar[dtype]] = inp + b * T * C3 + t * C3 + h * hs
                var preatt_bth: UnsafePointer[
                    Scalar[dtype]
                ] = preatt + b * NH * T * T + h * T * T + t * T
                var att_bth: UnsafePointer[
                    Scalar[dtype]
                ] = att + b * NH * T * T + h * T * T + t * T

                # pass 1: calculate query dot key and maxval
                var maxval: FLOAT = -10000.0  # TODO something better

                for t2 in range(t + 1):
                    var key_t2: UnsafePointer[
                        Scalar[dtype]
                    ] = inp + b * T * C3 + t2 * C3 + h * hs + C  # +C because it's key

                    # (query_t) dot (key_t2)
                    var val: FLOAT = 0.0

                    @parameter
                    fn _op[width: Int](iv: Int):
                        var t = query_t.load[width=width](iv) * key_t2.load[
                            width=width
                        ](iv)
                        val += t.reduce_add[1]()

                    vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=hs)

                    val *= scale
                    if val > maxval:
                        maxval = val

                    preatt_bth[t2] = val

                # pass 2: calculate the exp and keep track of sum
                var expsum: FLOAT = 0.0

                @parameter
                fn _op2[width: Int](iv: Int):
                    var expv = exp(preatt_bth.load[width=width](iv) - maxval)
                    expsum += expv.reduce_add[1]()
                    att_bth.store[width=width](iv, expv)

                vectorize[_op2, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=t + 1)

                var expsum_inv: FLOAT = 0.0
                if expsum != 0.0:
                    expsum_inv = FLOAT(1.0) / expsum

                # pass 3: normalize to get the softmax

                @parameter
                fn _op3[width: Int](t2: Int):
                    att_bth.store[width=width](
                        t2, att_bth.load[width=width](t2) * expsum_inv
                    )

                vectorize[_op3, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=t + 1)
                memset_zero(att_bth + t + 1, T - t - 1)

                # pass 4: accumulate weighted values into the output of attention
                var out_bth: UnsafePointer[Scalar[dtype]] = out + b * T * C + t * C + h * hs
                # for i in range(hs):
                #    out_bth[i] = 0.0
                memset_zero(out_bth, hs)

                for t2 in range(t + 1):
                    var value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2  # +C*2 because it's value
                    var att_btht2: FLOAT = att_bth[t2]

                    @parameter
                    fn _op4[width: Int](iv: Int):
                        out_bth.store[width=width](
                            iv,
                            out_bth.load[width=width](iv)
                            + att_btht2 * value_t2.load[width=width](iv),
                        )

                    vectorize[_op4, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=hs)

    parallelize[_calc](B)


fn attention_backward(
    dinp: UnsafePointer[Scalar[dtype]],
    dpreatt: UnsafePointer[Scalar[dtype]],
    datt: UnsafePointer[Scalar[dtype]],
    dout: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    att: UnsafePointer[Scalar[dtype]],
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
    var scale: FLOAT = FLOAT(1.0) / sqrt(hs)

    @parameter
    fn _calc(b: Int):
        for t in range(T):
            for h in range(NH):
                var att_bth: UnsafePointer[
                    Scalar[dtype]
                ] = att + b * NH * T * T + h * T * T + t * T
                var datt_bth: UnsafePointer[
                    Scalar[dtype]
                ] = datt + b * NH * T * T + h * T * T + t * T
                var dpreatt_bth: UnsafePointer[
                    Scalar[dtype]
                ] = dpreatt + b * NH * T * T + h * T * T + t * T
                var dquery_t: UnsafePointer[Scalar[dtype]] = dinp + b * T * C3 + t * C3 + h * hs
                var query_t: UnsafePointer[Scalar[dtype]] = inp + b * T * C3 + t * C3 + h * hs

                # backward pass 4, through the value accumulation
                var dout_bth: UnsafePointer[Scalar[dtype]] = dout + b * T * C + t * C + h * hs
                for t2 in range(t + 1):
                    var value_t2: UnsafePointer[
                        Scalar[dtype]
                    ] = inp + b * T * C3 + t2 * C3 + h * hs + C * 2  # +C*2 because it's value
                    var dvalue_t2: UnsafePointer[
                        Scalar[dtype]
                    ] = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2

                    @parameter
                    fn _op[width: Int](iv: Int):
                        # for i in range(hs):
                        # in the forward pass this was:
                        # out_bth[i] += att_bth[t2] * value_t2[i]
                        # so now we have:
                        datt_bth[t2] += (
                            value_t2.load[width=width](iv)
                            * dout_bth.load[width=width](iv)
                        ).reduce_add[1]()
                        dvalue_t2.store[width=width](
                            iv,
                            dvalue_t2.load[width=width](iv)
                            + att_bth[t2] * dout_bth.load[width=width](iv),
                        )

                    vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=hs)

                # backward pass 2 & 3, the softmax
                # note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in range(t + 1):

                    @parameter
                    fn _op3[width: Int](t3: Int):
                        var local_derivative = -att_bth[t2] * att_bth.load[width=width](
                            t3
                        )
                        dpreatt_bth.store[width=width](
                            t3,
                            dpreatt_bth.load[width=width](t3)
                            + local_derivative * datt_bth[t2],
                        )

                    vectorize[_op3, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=t + 1)
                    dpreatt_bth[t2] += att_bth[t2] * datt_bth[t2]

                # backward pass 1, the query @ key matmul
                for t2 in range(t + 1):
                    var key_t2: UnsafePointer[
                        Scalar[dtype]
                    ] = inp + b * T * C3 + t2 * C3 + h * hs + C  # +C because it's key
                    var dkey_t2: UnsafePointer[
                        Scalar[dtype]
                    ] = dinp + b * T * C3 + t2 * C3 + h * hs + C  # +C because it's key

                    @parameter
                    fn _op2[width: Int](iv: Int):
                        # for i in range(hs):
                        # in the forward pass this was:
                        # preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale
                        # so now we have:
                        dquery_t.store[width=width](
                            iv,
                            dquery_t.load[width=width](iv)
                            + key_t2.load[width=width](iv) * dpreatt_bth[t2] * scale,
                        )
                        dkey_t2.store[width=width](
                            iv,
                            dkey_t2.load[width=width](iv)
                            + query_t.load[width=width](iv) * dpreatt_bth[t2] * scale,
                        )

                    vectorize[_op2, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=hs)

    parallelize[_calc](B)


fn gelu_forward(out: UnsafePointer[Scalar[dtype]], inp: UnsafePointer[Scalar[dtype]], N: Int):
    var s: FLOAT = sqrt(2.0 / M_PI)

    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    fn _calc(ip: Int):
        @parameter
        fn _op[width: Int](_iv: Int):
            var iv = ip * num_vectorize + _iv
            
            var x = inp.load[width=width](iv)
            var cube = 0.044715 * pow(x, 3)
            out.store[width=width](iv, 0.5 * x * (1.0 + tanh(s * (x + cube))))

        vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize)

    parallelize[_calc](NUM_PARALLELIZE)


fn gelu_backward(
    dinp: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    dout: UnsafePointer[Scalar[dtype]],
    N: Int,
):
    var s: FLOAT = sqrt(2.0 / M_PI)
    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    fn _calc(ip: Int):
        @parameter
        fn _op[width: Int](_iv: Int):
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
            dinp.store[width=width](
                iv, dinp.load[width=width](iv) + local_grad * dout.load[width=width](iv)
            )

        vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize)

    parallelize[_calc](NUM_PARALLELIZE)


fn residual_forward(
    out: UnsafePointer[Scalar[dtype]],
    inp1: UnsafePointer[Scalar[dtype]],
    inp2: UnsafePointer[Scalar[dtype]],
    N: Int,
):
    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    fn _calc(ip: Int):
        @parameter
        fn _op[width: Int](_iv: Int):
            var iv = ip * num_vectorize + _iv
            out.store[width=width](
                iv, inp1.load[width=width](iv) + inp2.load[width=width](iv)
            )  # scale and shift it

        vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize)

    parallelize[_calc](NUM_PARALLELIZE)


fn residual_backward(
    dinp1: UnsafePointer[Scalar[dtype]],
    dinp2: UnsafePointer[Scalar[dtype]],
    dout: UnsafePointer[Scalar[dtype]],
    N: Int,
):
    var num_vectorize = N // NUM_PARALLELIZE

    @parameter
    fn _calc(ip: Int):
        @parameter
        fn _op[width: Int](_iv: Int):
            var iv = ip * num_vectorize + _iv

            dinp1.store[width=width](
                iv, dinp1.load[width=width](iv) + dout.load[width=width](iv)
            )  # scale and shift it
            dinp2.store[width=width](
                iv, dinp2.load[width=width](iv) + dout.load[width=width](iv)
            )  # scale and shift it

        vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize)

    parallelize[_calc](NUM_PARALLELIZE)


fn softmax_forward(
    probs: UnsafePointer[Scalar[dtype]], logits: UnsafePointer[Scalar[dtype]], B: Int, T: Int,V:Int, Vp: Int
):
    # output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    # input: logits is (B,T,Vp) of the unnormalized log probabilities
    # Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    # example: Vp is 50304 and V is 50257

    @parameter
    fn _calc(b: Int):
        # for b in range(B):
        for t in range(T):
            # probs <- softmax(logits)
            var logits_bt: UnsafePointer[Scalar[dtype]] = logits + b * T * Vp + t * Vp
            var probs_bt: UnsafePointer[Scalar[dtype]] = probs + b * T * Vp + t * Vp

            var maxval: FLOAT = -10000.0  # TODO something better
            for i in range(V):
                if logits_bt[i] > maxval:
                    maxval = logits_bt[i]

            var sum: FLOAT = 0.0

            @parameter
            fn _op[width: Int](iv: Int):
                probs_bt.store[width=width](
                    iv, exp(logits_bt.load[width=width](iv) - maxval)
                )
                sum += probs_bt.load[width=width](iv).reduce_add[1]()

            vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=V)

            @parameter
            fn _op2[width: Int](iv: Int):
                probs_bt.store[width=width](
                    iv, probs_bt.load[width=width](iv) / sum
                )  # scale and shift it

            vectorize[_op2, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=V)


            # for extra super safety we may wish to include this too,
            # forcing the probabilities here to be zero, but it shouldn't matter
            
            @parameter
            fn _op3[width: Int](iv: Int):
                probs_bt.store[width=width](
                    iv+V,0.0
                ) 
            vectorize[_op3, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=Vp-V)

            
           

    parallelize[_calc](B)


fn crossentropy_forward(
    losses: UnsafePointer[Scalar[dtype]],
    probs: UnsafePointer[Scalar[dtype]],
    targets: UnsafePointer[Scalar[dtype_int]],
    B: Int,
    T: Int,
    Vp: Int
):
    # output: losses is (B,T) of the individual losses at each position
    # input: probs are (B,T,Vp) of the probabilities
    # input: targets is (B,T) of integers giving the correct index in logits

    @parameter
    fn _calc(b: Int):
        for t in range(T):  # todo
            # loss = -log(probs[target])
            var probs_bt: UnsafePointer[Scalar[dtype]] = probs + b * T * Vp + t * Vp
            var ix = targets[b * T + t]
            losses[b * T + t] = -log(probs_bt.load(ix))

    parallelize[_calc](B)


fn crossentropy_softmax_backward(
    dlogits: UnsafePointer[Scalar[dtype]],
    dlosses: UnsafePointer[Scalar[dtype]],
    probs: UnsafePointer[Scalar[dtype]],
    targets: UnsafePointer[Scalar[dtype_int]],
    B: Int,
    T: Int,
    V: Int,
    Vp: Int
):
    # backwards through both softmax and crossentropy

    @parameter
    fn _calc(b: Int):
        for t in range(T):
            var dlogits_bt: UnsafePointer[Scalar[dtype]] = dlogits + b * T * Vp + t * Vp
            var probs_bt: UnsafePointer[Scalar[dtype]] = probs + b * T * Vp + t * Vp
            var dloss: FLOAT = dlosses[b * T + t]
            var ix = targets[b * T + t]

            @parameter
            fn _op[width: Int](iv: Int):
                dlogits_bt.store[width=width](
                    iv,
                    dlogits_bt.load[width=width](iv)
                    + probs_bt.load[width=width](iv) * dloss,
                )

            vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](size=V)

            if ix >= 0 and ix < V:
                dlogits_bt.store(ix,dlogits_bt.load(ix)-dloss)

    parallelize[_calc](B)


# ----------------------------------------------------------------------------
# GPT-2 model definition

# the parameters of the model

alias NUM_PARAMETER_TENSORS = 16


struct ParameterTensors:
    var params_memory: UnsafePointer[Scalar[dtype]]

    var wte: UnsafePointer[Scalar[dtype]]  # (V, C)
    var wpe: UnsafePointer[Scalar[dtype]]  # (maxT, C)
    var ln1w: UnsafePointer[Scalar[dtype]]  # (L, C)
    var ln1b: UnsafePointer[Scalar[dtype]]  # (L, C)
    var qkvw: UnsafePointer[Scalar[dtype]]  # (L, 3*C, C)
    var qkvb: UnsafePointer[Scalar[dtype]]  # (L, 3*C)
    var attprojw: UnsafePointer[Scalar[dtype]]  # (L, C, C)
    var attprojb: UnsafePointer[Scalar[dtype]]  # (L, C)
    var ln2w: UnsafePointer[Scalar[dtype]]  # (L, C)
    var ln2b: UnsafePointer[Scalar[dtype]]  # (L, C)
    var fcw: UnsafePointer[Scalar[dtype]]  # (L, 4*C, C)
    var fcb: UnsafePointer[Scalar[dtype]]  # (L, 4*C)
    var fcprojw: UnsafePointer[Scalar[dtype]]  # (L, C, 4*C)
    var fcprojb: UnsafePointer[Scalar[dtype]]  # (L, C)
    var lnfw: UnsafePointer[Scalar[dtype]]  # (C)
    var lnfb: UnsafePointer[Scalar[dtype]]  # (C)

    fn __init__(
        inout self
    ):
        self.params_memory = UnsafePointer[Scalar[dtype]]()

        self.wte = UnsafePointer[Scalar[dtype]]()
        self.wpe = UnsafePointer[Scalar[dtype]]()
        self.ln1w = UnsafePointer[Scalar[dtype]]()
        self.ln1b = UnsafePointer[Scalar[dtype]]()
        self.qkvw = UnsafePointer[Scalar[dtype]]()
        self.qkvb = UnsafePointer[Scalar[dtype]]()
        self.attprojw = UnsafePointer[Scalar[dtype]]()
        self.attprojb = UnsafePointer[Scalar[dtype]]()
        self.ln2w = UnsafePointer[Scalar[dtype]]()
        self.ln2b = UnsafePointer[Scalar[dtype]]()
        self.fcw = UnsafePointer[Scalar[dtype]]()
        self.fcb = UnsafePointer[Scalar[dtype]]()
        self.fcprojw = UnsafePointer[Scalar[dtype]]()
        self.fcprojb = UnsafePointer[Scalar[dtype]]()
        self.lnfw = UnsafePointer[Scalar[dtype]]()
        self.lnfb = UnsafePointer[Scalar[dtype]]()

    fn alloc_and_point_parameters(
        inout self,
        param_sizes: InlinedFixedVector[type=Int, size=NUM_PARAMETER_TENSORS],
    ) -> UnsafePointer[Scalar[dtype]]:
        var num_parameters: Int = 0

        for i in range(NUM_PARAMETER_TENSORS):
            num_parameters += param_sizes[i]

        # malloc all parameters all at once
        self.params_memory = UnsafePointer[Scalar[dtype]]().alloc(num_parameters)
        # assign all the tensors

        var ptrs = List(
            UnsafePointer.address_of(self.wte),
            UnsafePointer.address_of(self.wpe),
            UnsafePointer.address_of(self.ln1w),
            UnsafePointer.address_of(self.ln1b),
            UnsafePointer.address_of(self.qkvw),
            UnsafePointer.address_of(self.qkvb),
            UnsafePointer.address_of(self.attprojw),
            UnsafePointer.address_of(self.attprojb),
            UnsafePointer.address_of(self.ln2w),
            UnsafePointer.address_of(self.ln2b),
            UnsafePointer.address_of(self.fcw),
            UnsafePointer.address_of(self.fcb),
            UnsafePointer.address_of(self.fcprojw),
            UnsafePointer.address_of(self.fcprojb),
            UnsafePointer.address_of(self.lnfw),
            UnsafePointer.address_of(self.lnfb),
        )

        var params_memory_iterator: UnsafePointer[Scalar[dtype]] = self.params_memory

        for i in range(NUM_PARAMETER_TENSORS):
            ptrs[i][] = params_memory_iterator
            params_memory_iterator += param_sizes[i]

        return self.params_memory


alias NUM_ACTIVATION_TENSORS = 23


@value
struct ActivationTensors:
    var encoded: UnsafePointer[Scalar[dtype]]  # (B, T, C)
    var ln1: UnsafePointer[Scalar[dtype]]  # (L, B, T, C)
    var ln1_mean: UnsafePointer[Scalar[dtype]]  # (L, B, T)
    var ln1_rstd: UnsafePointer[Scalar[dtype]]  # (L, B, T)
    var qkv: UnsafePointer[Scalar[dtype]]  # (L, B, T, 3*C)
    var atty: UnsafePointer[Scalar[dtype]]  # (L, B, T, C)
    var preatt: UnsafePointer[Scalar[dtype]]  # (L, B, NH, T, T)
    var att: UnsafePointer[Scalar[dtype]]  # (L, B, NH, T, T)
    var attproj: UnsafePointer[Scalar[dtype]]  # (L, B, T, C)
    var residual2: UnsafePointer[Scalar[dtype]]  # (L, B, T, C)
    var ln2: UnsafePointer[Scalar[dtype]]  # (L, B, T, C)
    var ln2_mean: UnsafePointer[Scalar[dtype]]  # (L, B, T)
    var ln2_rstd: UnsafePointer[Scalar[dtype]]  # (L, B, T)
    var fch: UnsafePointer[Scalar[dtype]]  # (L, B, T, 4*C)
    var fch_gelu: UnsafePointer[Scalar[dtype]]  # (L, B, T, 4*C)
    var fcproj: UnsafePointer[Scalar[dtype]]  # (L, B, T, C)
    var residual3: UnsafePointer[Scalar[dtype]]  # (L, B, T, C)
    var lnf: UnsafePointer[Scalar[dtype]]  # (B, T, C)
    var lnf_mean: UnsafePointer[Scalar[dtype]]  # (B, T)
    var lnf_rstd: UnsafePointer[Scalar[dtype]]  # (B, T)
    var logits: UnsafePointer[Scalar[dtype]]  # (B, T, V)
    var probs: UnsafePointer[Scalar[dtype]]  # (B, T, V)
    var losses: UnsafePointer[Scalar[dtype]]  # (B, T)

    fn __init__(
        inout self,
    ):
        self.encoded = UnsafePointer[Scalar[dtype]]()
        self.ln1 = UnsafePointer[Scalar[dtype]]()
        self.ln1_mean = UnsafePointer[Scalar[dtype]]()
        self.ln1_rstd = UnsafePointer[Scalar[dtype]]()
        self.qkv = UnsafePointer[Scalar[dtype]]()
        self.atty = UnsafePointer[Scalar[dtype]]()
        self.preatt = UnsafePointer[Scalar[dtype]]()
        self.att = UnsafePointer[Scalar[dtype]]()
        self.attproj = UnsafePointer[Scalar[dtype]]()
        self.residual2 = UnsafePointer[Scalar[dtype]]()
        self.ln2 = UnsafePointer[Scalar[dtype]]()
        self.ln2_mean = UnsafePointer[Scalar[dtype]]()
        self.ln2_rstd = UnsafePointer[Scalar[dtype]]()
        self.fch = UnsafePointer[Scalar[dtype]]()
        self.fch_gelu = UnsafePointer[Scalar[dtype]]()
        self.fcproj = UnsafePointer[Scalar[dtype]]()
        self.residual3 = UnsafePointer[Scalar[dtype]]()
        self.lnf = UnsafePointer[Scalar[dtype]]()
        self.lnf_mean = UnsafePointer[Scalar[dtype]]()
        self.lnf_rstd = UnsafePointer[Scalar[dtype]]()
        self.logits = UnsafePointer[Scalar[dtype]]()
        self.probs = UnsafePointer[Scalar[dtype]]()
        self.losses = UnsafePointer[Scalar[dtype]]()

    fn alloc_and_point_activations(
        inout self, act_sizes: InlinedFixedVector[type=Int, size=NUM_ACTIVATION_TENSORS]
    ) -> UnsafePointer[Scalar[dtype]]:
        var ptrs = List(
            UnsafePointer.address_of(self.encoded),
            UnsafePointer.address_of(self.ln1),
            UnsafePointer.address_of(self.ln1_mean),
            UnsafePointer.address_of(self.ln1_rstd),
            UnsafePointer.address_of(self.qkv),
            UnsafePointer.address_of(self.atty),
            UnsafePointer.address_of(self.preatt),
            UnsafePointer.address_of(self.att),
            UnsafePointer.address_of(self.attproj),
            UnsafePointer.address_of(self.residual2),
            UnsafePointer.address_of(self.ln2),
            UnsafePointer.address_of(self.ln2_mean),
            UnsafePointer.address_of(self.ln2_rstd),
            UnsafePointer.address_of(self.fch),
            UnsafePointer.address_of(self.fch_gelu),
            UnsafePointer.address_of(self.fcproj),
            UnsafePointer.address_of(self.residual3),
            UnsafePointer.address_of(self.lnf),
            UnsafePointer.address_of(self.lnf_mean),
            UnsafePointer.address_of(self.lnf_rstd),
            UnsafePointer.address_of(self.logits),
            UnsafePointer.address_of(self.probs),
            UnsafePointer.address_of(self.losses),
        )

        var num_activations: Int = 0

        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += act_sizes[i]

        var acts_memory = UnsafePointer[Scalar[dtype]]().alloc(num_activations)

        var acts_memory_iterator: UnsafePointer[Scalar[dtype]] = acts_memory
        for i in range(NUM_ACTIVATION_TENSORS):
            ptrs[i][] = acts_memory_iterator
            acts_memory_iterator += act_sizes[i]

        return acts_memory


@value
struct GPT2Config:
    var max_seq_len: Int  # max sequence length, e.g. 1024
    var vocab_size: Int  # vocab size, e.g. 50257
    var num_layers: Int  # number of layers, e.g. 12
    var num_heads: Int  # number of heads in attention, e.g. 12
    var channels: Int  # number of channels, e.g. 768
    var padded_vocab_size:Int # padded to e.g. %128==0, 50304


struct GPT2:
    var config: GPT2Config
    # the weights of the model, and their sizes
    var params: ParameterTensors
    var param_sizes: InlinedFixedVector[type=Int, size=NUM_PARAMETER_TENSORS]
    var params_memory: UnsafePointer[Scalar[dtype]]
    var num_parameters: Int
    # gradients of the weights
    var grads: ParameterTensors
    var grads_memory: UnsafePointer[Scalar[dtype]]
    # buffers for the AdamW optimizer
    var m_memory: UnsafePointer[Scalar[dtype]]
    var v_memory: UnsafePointer[Scalar[dtype]]
    # the activations of the model, and their sizes
    var acts: ActivationTensors
    var act_sizes: InlinedFixedVector[type=Int, size=NUM_ACTIVATION_TENSORS]
    var acts_memory: UnsafePointer[Scalar[dtype]]
    var num_activations: Int
    # gradients of the activations
    var grads_acts: ActivationTensors
    var grads_acts_memory: UnsafePointer[Scalar[dtype]]
    # other run state configuration
    var batch_size: Int  # the batch size (B) of current forward pass
    var seq_len: Int  # the sequence length (T) of current forward pass
    var inputs: UnsafePointer[Scalar[dtype_int]]  # the input tokens for the current forward pass
    var targets: UnsafePointer[
        Scalar[dtype_int]
    ]  # the target tokens for the current forward pass
    var mean_loss: FLOAT  # after a forward pass with targets, will be populated with the mean loss
    var checkpoint_path: StringRef

    fn __init__(inout self, checkpoint_path: StringRef) raises:
        self.checkpoint_path = checkpoint_path

        self.param_sizes = InlinedFixedVector[type=Int, size=NUM_PARAMETER_TENSORS](
            NUM_PARAMETER_TENSORS
        )
        self.act_sizes = InlinedFixedVector[type=Int, size=NUM_ACTIVATION_TENSORS](
            NUM_ACTIVATION_TENSORS
        )

        var model_file = open(checkpoint_path, "r")

        var model_header = UnsafePointer[Scalar[dtype.int32]].alloc(256)
        read_to_dtype_pointer[DType.int32](model_header,model_file,256)

        if model_header[0] != 20240326:
            print("Bad magic model file",model_header[0])
            exit(1)
        if model_header[1] != 3:
            print("Bad version in model file")
            exit(1)

        # read in hyperparameters

        self.config = GPT2Config(
            int(model_header[2]),
            int(model_header[3]),
            int(model_header[4]),
            int(model_header[5]),
            int(model_header[6]),
            int(model_header[7]),
        )
       
        var maxT: Int = self.config.max_seq_len
        var V: Int = self.config.vocab_size
        var L: Int = self.config.num_layers
        var NH: Int = self.config.num_heads
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
        self.params = ParameterTensors()
        self.params_memory = self.params.alloc_and_point_parameters(self.param_sizes)

        read_to_dtype_pointer[DType.float32](self.params_memory,model_file,num_parameters)
        model_file.close() 
        
        # other inits
        self.acts = ActivationTensors()
        self.num_activations = 0  # for now

        self.acts_memory = NULL
        self.grads_memory = NULL
        self.m_memory = NULL
        self.v_memory = NULL
        self.grads_acts_memory = NULL
        self.inputs = NULL_INT
        self.targets = NULL_INT
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0  # -1.0 will designate no loss

        self.grads = ParameterTensors()
        self.grads_acts = ActivationTensors()

        print("[GPT-2]")
        print("max_seq_len:", self.config.max_seq_len)
        print("vocab_size:", self.config.vocab_size)
        print("padded_vocab_size:", self.config.padded_vocab_size)
        print("num_layers:", self.config.num_layers)
        print("num_heads:", self.config.num_heads)
        print("channels:", self.config.channels)

        print("num_parameters:", num_parameters)



fn gpt2_forward(
    inout model: GPT2,
    inputs: UnsafePointer[Scalar[dtype_int]],
    targets: UnsafePointer[Scalar[dtype_int]],
    B: Int,
    T: Int,
):
    # targets are optional and could be NULL

    # ensure the model was initialized or error out
    if model.params_memory == NULL:
        print("Error: model was not initialized properly.")

    # convenience parameters
    var V: Int = model.config.vocab_size
    var Vp: Int = model.config.padded_vocab_size
    var L: Int = model.config.num_layers
    var NH: Int = model.config.num_heads
    var C: Int = model.config.channels

    # allocate space for all the activations if needed (done here, lazily)
    if model.acts_memory == NULL:
        # record the current B,T as well
        model.batch_size = B
        model.seq_len = T

        # and now allocate the space
        model.act_sizes[0] = B * T * C
        model.act_sizes[1] = L * B * T * C
        model.act_sizes[2] = L * B * T
        model.act_sizes[3] = L * B * T
        model.act_sizes[4] = L * B * T * 3 * C
        model.act_sizes[5] = L * B * T * C
        model.act_sizes[6] = L * B * NH * T * T
        model.act_sizes[7] = L * B * NH * T * T
        model.act_sizes[8] = L * B * T * C
        model.act_sizes[9] = L * B * T * C
        model.act_sizes[10] = L * B * T * C
        model.act_sizes[11] = L * B * T
        model.act_sizes[12] = L * B * T
        model.act_sizes[13] = L * B * T * 4 * C
        model.act_sizes[14] = L * B * T * 4 * C
        model.act_sizes[15] = L * B * T * C
        model.act_sizes[16] = L * B * T * C
        model.act_sizes[17] = B * T * C
        model.act_sizes[18] = B * T
        model.act_sizes[19] = B * T
        model.act_sizes[20] = B * T * Vp
        model.act_sizes[21] = B * T * Vp
        model.act_sizes[22] = B * T

        var num_activations: Int = 0
        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += model.act_sizes[i]

        print("num_activations:", num_activations)

        model.acts_memory = model.acts.alloc_and_point_activations(model.act_sizes)
        model.num_activations = num_activations
        # also create memory for caching inputs and targets

        model.inputs = UnsafePointer[Scalar[dtype_int]]().alloc(B * T)
        model.targets = UnsafePointer[Scalar[dtype_int]]().alloc(B * T)

    else:
        # validate B,T is no larger than what was previously allocated
        # in principle, we could re-allocate a larger chunk of memory, for now we just error out
        if B > model.batch_size or T > model.seq_len:
            print("Error: batch size or sequence length is inadequately large")
            # print("Model: B=%d T=%d, Desired: B=%d T=%d\n", model.batch_size, model.seq_len, B, T)

    # cache the inputs/targets
    memcpy(model.inputs, inputs, B * T)

    if targets != NULL_INT:
        memcpy(model.targets, targets, B * T)

    # forward pass

    var residual: UnsafePointer[Scalar[dtype]]
    encoder_forward(
        model.acts.encoded, inputs, model.params.wte, model.params.wpe, B, T, C
    )  # encoding goes into residual[0]

    for l in range(L):
        residual = model.acts.residual3 + (l - 1) * B * T * C

        if l == 0:
            residual = model.acts.encoded

        # get the pointers of the weights for this layer
        var l_ln1w: UnsafePointer[Scalar[dtype]] = model.params.ln1w + l * C
        var l_ln1b: UnsafePointer[Scalar[dtype]] = model.params.ln1b + l * C
        var l_qkvw: UnsafePointer[Scalar[dtype]] = model.params.qkvw + l * 3 * C * C
        var l_qkvb: UnsafePointer[Scalar[dtype]] = model.params.qkvb + l * 3 * C
        var l_attprojw: UnsafePointer[Scalar[dtype]] = model.params.attprojw + l * C * C
        var l_attprojb: UnsafePointer[Scalar[dtype]] = model.params.attprojb + l * C
        var l_ln2w: UnsafePointer[Scalar[dtype]] = model.params.ln2w + l * C
        var l_ln2b: UnsafePointer[Scalar[dtype]] = model.params.ln2b + l * C
        var l_fcw: UnsafePointer[Scalar[dtype]] = model.params.fcw + l * 4 * C * C
        var l_fcb: UnsafePointer[Scalar[dtype]] = model.params.fcb + l * 4 * C
        var l_fcprojw: UnsafePointer[Scalar[dtype]] = model.params.fcprojw + l * C * 4 * C
        var l_fcprojb: UnsafePointer[Scalar[dtype]] = model.params.fcprojb + l * C

        # get the pointers of the activations for this layer
        var l_ln1: UnsafePointer[Scalar[dtype]] = model.acts.ln1 + l * B * T * C
        var l_ln1_mean: UnsafePointer[Scalar[dtype]] = model.acts.ln1_mean + l * B * T
        var l_ln1_rstd: UnsafePointer[Scalar[dtype]] = model.acts.ln1_rstd + l * B * T
        var l_qkv: UnsafePointer[Scalar[dtype]] = model.acts.qkv + l * B * T * 3 * C
        var l_atty: UnsafePointer[Scalar[dtype]] = model.acts.atty + l * B * T * C
        var l_preatt: UnsafePointer[Scalar[dtype]] = model.acts.preatt + l * B * NH * T * T
        var l_att: UnsafePointer[Scalar[dtype]] = model.acts.att + l * B * NH * T * T
        var l_attproj: UnsafePointer[Scalar[dtype]] = model.acts.attproj + l * B * T * C
        var l_residual2: UnsafePointer[Scalar[dtype]] = model.acts.residual2 + l * B * T * C
        var l_ln2: UnsafePointer[Scalar[dtype]] = model.acts.ln2 + l * B * T * C
        var l_ln2_mean: UnsafePointer[Scalar[dtype]] = model.acts.ln2_mean + l * B * T
        var l_ln2_rstd: UnsafePointer[Scalar[dtype]] = model.acts.ln2_rstd + l * B * T
        var l_fch: UnsafePointer[Scalar[dtype]] = model.acts.fch + l * B * T * 4 * C
        var l_fch_gelu: UnsafePointer[Scalar[dtype]] = model.acts.fch_gelu + l * B * T * 4 * C
        var l_fcproj: UnsafePointer[Scalar[dtype]] = model.acts.fcproj + l * B * T * C
        var l_residual3: UnsafePointer[Scalar[dtype]] = model.acts.residual3 + l * B * T * C

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
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C)
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C)

    residual = (
        model.acts.residual3 + (L - 1) * B * T * C
    )  # last residual is in residual3
    layernorm_forward(
        model.acts.lnf,
        model.acts.lnf_mean,
        model.acts.lnf_rstd,
        residual,
        model.params.lnfw,
        model.params.lnfb,
        B,
        T,
        C,
    )
    matmul_forward(
        model.acts.logits, model.acts.lnf, model.params.wte, NULL, B, T, C, Vp
    )
    softmax_forward(model.acts.probs, model.acts.logits, B, T, V,Vp)

    # also forward the cross-entropy loss function if we have the targets
    if targets != NULL_INT:
        crossentropy_forward(model.acts.losses, model.acts.probs, targets, B, T, Vp)
        # for convenience also evaluate the mean loss
        var mean_loss: FLOAT = 0.0
        for i in range(B * T):
            mean_loss += model.acts.losses[i]
        mean_loss /= B * T
        model.mean_loss = mean_loss
    else:
        # if we don't have targets, we don't have a loss
        model.mean_loss = -1.0


fn gpt2_zero_grad(inout model: GPT2):
    if model.grads_memory != NULL:
        memset_zero(model.grads_memory, model.num_parameters)

    if model.grads_acts_memory != NULL:
        memset_zero(model.grads_acts_memory, model.num_activations)


fn gpt2_backward(inout model: GPT2):
    # double check we forwarded previously, with targets
    if model.mean_loss == -1.0:
        print("Error: must forward with targets before backward\n")

    # lazily allocate the memory for gradients of the weights and activations, if needed
    if model.grads_memory == NULL:
        model.grads_memory = model.grads.alloc_and_point_parameters(model.param_sizes)
        model.grads_acts_memory = model.grads_acts.alloc_and_point_activations(
            model.act_sizes
        )
        gpt2_zero_grad(model)

    # convenience shortcuts
    var B: Int = model.batch_size
    var T: Int = model.seq_len
    var V: Int = model.config.vocab_size
    var Vp: Int = model.config.padded_vocab_size
    var L: Int = model.config.num_layers
    var NH: Int = model.config.num_heads
    var C: Int = model.config.channels

    # backward pass

    # we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    # technically this is a small, inline backward() pass of calculating
    # total, final loss as the mean over all losses over all (B,T) positions in the batch
  
    
    var dloss_mean: FLOAT = FLOAT(1.0) / (B * T)

    @parameter
    fn _op[width: Int](iv: Int):
        model.grads_acts.losses.store[width=width](iv, dloss_mean)

    vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR]((B * T))

    crossentropy_softmax_backward(
        model.grads_acts.logits,
        model.grads_acts.losses,
        model.acts.probs,
        model.targets,
        B,
        T,
        V,
        Vp
    )
    matmul_backward(
        model.grads_acts.lnf,
        model.grads.wte,
        NULL,
        model.grads_acts.logits,
        model.acts.lnf,
        model.params.wte,
        B,
        T,
        C,
        Vp,
    )
    var residual: UnsafePointer[Scalar[dtype]] = model.acts.residual3 + (
        L - 1
    ) * B * T * C  # last layer's residual
    var dresidual: UnsafePointer[Scalar[dtype]] = model.grads_acts.residual3 + (
        L - 1
    ) * B * T * C  # write to last layer's residual
    layernorm_backward(
        dresidual,
        model.grads.lnfw,
        model.grads.lnfb,
        model.grads_acts.lnf,
        residual,
        model.params.lnfw,
        model.acts.lnf_mean,
        model.acts.lnf_rstd,
        B,
        T,
        C,
    )

    for l in range(L - 1, -1, -1):
        var residual = model.acts.encoded
        var dresidual = model.grads_acts.encoded
        if l != 0:
            residual = model.acts.residual3 + (l - 1) * B * T * C
            dresidual = model.grads_acts.residual3 + (l - 1) * B * T * C

        # get the pointers of the weights for this layer
        var l_ln1w: UnsafePointer[Scalar[dtype]] = model.params.ln1w + l * C
        var l_qkvw: UnsafePointer[Scalar[dtype]] = model.params.qkvw + l * 3 * C * C
        var l_attprojw: UnsafePointer[Scalar[dtype]] = model.params.attprojw + l * C * C
        var l_ln2w: UnsafePointer[Scalar[dtype]] = model.params.ln2w + l * C
        var l_fcw: UnsafePointer[Scalar[dtype]] = model.params.fcw + l * 4 * C * C
        var l_fcprojw: UnsafePointer[Scalar[dtype]] = model.params.fcprojw + l * C * 4 * C
        # get the pointers of the gradients of the weights for this layer
        var dl_ln1w: UnsafePointer[Scalar[dtype]] = model.grads.ln1w + l * C
        var dl_ln1b: UnsafePointer[Scalar[dtype]] = model.grads.ln1b + l * C
        var dl_qkvw: UnsafePointer[Scalar[dtype]] = model.grads.qkvw + l * 3 * C * C
        var dl_qkvb: UnsafePointer[Scalar[dtype]] = model.grads.qkvb + l * 3 * C
        var dl_attprojw: UnsafePointer[Scalar[dtype]] = model.grads.attprojw + l * C * C
        var dl_attprojb: UnsafePointer[Scalar[dtype]] = model.grads.attprojb + l * C
        var dl_ln2w: UnsafePointer[Scalar[dtype]] = model.grads.ln2w + l * C
        var dl_ln2b: UnsafePointer[Scalar[dtype]] = model.grads.ln2b + l * C
        var dl_fcw: UnsafePointer[Scalar[dtype]] = model.grads.fcw + l * 4 * C * C
        var dl_fcb: UnsafePointer[Scalar[dtype]] = model.grads.fcb + l * 4 * C
        var dl_fcprojw: UnsafePointer[Scalar[dtype]] = model.grads.fcprojw + l * C * 4 * C
        var dl_fcprojb: UnsafePointer[Scalar[dtype]] = model.grads.fcprojb + l * C
        # get the pointers of the activations for this layer
        var l_ln1: UnsafePointer[Scalar[dtype]] = model.acts.ln1 + l * B * T * C
        var l_ln1_mean: UnsafePointer[Scalar[dtype]] = model.acts.ln1_mean + l * B * T
        var l_ln1_rstd: UnsafePointer[Scalar[dtype]] = model.acts.ln1_rstd + l * B * T
        var l_qkv: UnsafePointer[Scalar[dtype]] = model.acts.qkv + l * B * T * 3 * C
        var l_atty: UnsafePointer[Scalar[dtype]] = model.acts.atty + l * B * T * C
        var l_att: UnsafePointer[Scalar[dtype]] = model.acts.att + l * B * NH * T * T
        var l_residual2: UnsafePointer[Scalar[dtype]] = model.acts.residual2 + l * B * T * C
        var l_ln2: UnsafePointer[Scalar[dtype]] = model.acts.ln2 + l * B * T * C
        var l_ln2_mean: UnsafePointer[Scalar[dtype]] = model.acts.ln2_mean + l * B * T
        var l_ln2_rstd: UnsafePointer[Scalar[dtype]] = model.acts.ln2_rstd + l * B * T
        var l_fch: UnsafePointer[Scalar[dtype]] = model.acts.fch + l * B * T * 4 * C
        var l_fch_gelu: UnsafePointer[Scalar[dtype]] = model.acts.fch_gelu + l * B * T * 4 * C
        # get the pointers of the gradients of the activations for this layer
        var dl_ln1: UnsafePointer[Scalar[dtype]] = model.grads_acts.ln1 + l * B * T * C
        var dl_qkv: UnsafePointer[Scalar[dtype]] = model.grads_acts.qkv + l * B * T * 3 * C
        var dl_atty: UnsafePointer[Scalar[dtype]] = model.grads_acts.atty + l * B * T * C
        var dl_preatt: UnsafePointer[
            Scalar[dtype]
        ] = model.grads_acts.preatt + l * B * NH * T * T
        var dl_att: UnsafePointer[Scalar[dtype]] = model.grads_acts.att + l * B * NH * T * T
        var dl_attproj: UnsafePointer[Scalar[dtype]] = model.grads_acts.attproj + l * B * T * C
        var dl_residual2: UnsafePointer[
            Scalar[dtype]
        ] = model.grads_acts.residual2 + l * B * T * C
        var dl_ln2: UnsafePointer[Scalar[dtype]] = model.grads_acts.ln2 + l * B * T * C
        var dl_fch: UnsafePointer[Scalar[dtype]] = model.grads_acts.fch + l * B * T * 4 * C
        var dl_fch_gelu: UnsafePointer[
            Scalar[dtype]
        ] = model.grads_acts.fch_gelu + l * B * T * 4 * C
        var dl_fcproj: UnsafePointer[Scalar[dtype]] = model.grads_acts.fcproj + l * B * T * C
        var dl_residual3: UnsafePointer[
            Scalar[dtype]
        ] = model.grads_acts.residual3 + l * B * T * C

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
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C)
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
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C)
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
        model.grads.wte,
        model.grads.wpe,
        model.grads_acts.encoded,
        model.inputs,
        B,
        T,
        C,
    )


fn gpt2_update(
    inout model: GPT2,
    learning_rate: FLOAT,
    beta1: FLOAT,
    beta2: FLOAT,
    eps: FLOAT,
    weight_decay: FLOAT,
    t: Int,
):
    # reference: https:#pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    # lazily allocate the memory for m_memory and v_memory
    if model.m_memory == NULL:
        model.m_memory = UnsafePointer[Scalar[dtype]]().alloc(model.num_parameters)
        model.v_memory = UnsafePointer[Scalar[dtype]]().alloc(model.num_parameters)

        memset_zero(model.m_memory, model.num_parameters)
        memset_zero(model.v_memory, model.num_parameters)

    var num_vectorize = model.num_parameters // NUM_PARALLELIZE

    @parameter
    fn _calc(ip: Int):
        @parameter
        fn _op[width: Int](_iv: Int):
            var iv = ip * num_vectorize + _iv
            var param = model.params_memory.load[width=width](iv)
            var grad = model.grads_memory.load[width=width](iv)

            # update the first moment (momentum)
            var m = beta1 * model.m_memory.load[width=width](iv) + (1.0 - beta1) * grad
            # update the second moment (RMSprop)
            var v = beta2 * model.v_memory.load[width=width](iv) + (
                1.0 - beta2
            ) * grad * grad
            # bias-correct both moments
            var m_hat = m / (1.0 - pow(beta1, t))
            var v_hat = v / (1.0 - pow(beta2, t))

            # update
            model.m_memory.store[width=width](iv, m)
            model.v_memory.store[width=width](iv, v)
            model.params_memory.store[width=width](
                iv,
                model.params_memory.load[width=width](iv)
                - learning_rate * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param),
            )

        vectorize[_op, SIMD_WIDTH, unroll_factor=UNROLL_FACTOR](num_vectorize)

    parallelize[_calc](NUM_PARALLELIZE)


fn gpt2_free(inout model: GPT2):
    model.params_memory.free()
    model.grads_memory.free()
    model.m_memory.free()
    model.v_memory.free()
    model.acts_memory.free()
    model.grads_acts_memory.free()
    model.inputs.free()
    model.targets.free()


# ifndef TESTING
# if we are TESTING (see test_gpt2.c), we'll skip the maiN:Int32 below

# ----------------------------------------------------------------------------
# data loader lite
# returns random batches of data from a file of integers


struct DataLoader:
    # hyperparameters
    var B: Int
    var T: Int
    # input handling and its state
    var filename: StringRef
    var tokens_file: FileHandle
    var file_size: Int
    var current_position: Int
    # output memory
    var batch: UnsafePointer[Scalar[dtype_int]]
    var inputs: UnsafePointer[Scalar[dtype_int]]
    var targets: UnsafePointer[Scalar[dtype_int]]
    # convenience variables
    var num_batches: Int

    fn __init__(inout self):
        self.B = 0
        self.T = 0
        self.filename = ""
        self.tokens_file = FileHandle()
        self.file_size = 0
        self.current_position = 0
        self.batch = UnsafePointer[Scalar[dtype_int]]()
        self.inputs = UnsafePointer[Scalar[dtype_int]]()
        self.targets = UnsafePointer[Scalar[dtype_int]]()
        self.num_batches = 0


fn dataloader_init(
    inout loader: DataLoader, filename: StringRef, B: Int, T: Int
) raises:
    loader.B = B
    loader.T = T
    try:
        loader.tokens_file = open(filename, "rb")
    except e:
        print("Error opening file",filename,e)
        exit(1)
      
    # determine the file size
    var _os = Python.import_module("os")
    loader.file_size = int(_os.path.getsize(filename))

    if loader.file_size < (B * T + 1) * 4:
        print("Error: file size is too small for the batch size and sequence length\n")

    loader.current_position = 0  # start at the beginning

    # allocate space for B*T + 1 integers to store the inputs and targets loader.batch = (int*) malloc((B * T + 1) * sizeof(int))

    loader.batch = UnsafePointer[Scalar[dtype_int]]().alloc(B * T + 1)
    loader.inputs = loader.batch
    loader.targets = loader.batch + 1  # targets are shifted by one
    loader.num_batches = loader.file_size // (B * T * SIZEOF_INT)

    

fn dataloader_reset(inout loader: DataLoader):
    loader.current_position = 0


fn dataloader_next_batch(inout loader: DataLoader) raises:
    var B: Int = loader.B
    var T: Int = loader.T

    # if we are at the end of the file, loop back to the beginning
    if loader.current_position + ((B * T + 1) * SIZEOF_INT) > loader.file_size:
        loader.current_position = 0

    # read the B*T+1 integers from the file into batch
    var q = loader.tokens_file.seek(loader.current_position)

    read_to_dtype_pointer(loader.batch,loader.tokens_file,B * T + 1)

    # advance the current position by B*T integers
    loader.current_position += B * T * SIZEOF_INT


fn dataloader_free(inout loader: DataLoader) raises:
    loader.tokens_file.close()
    loader.batch.free()


# ----------------------------------------------------------------------------
# sampler


fn random_u32(inout state: UInt64) -> UInt32:
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    return ((state * RU32_HEX) >> 32).cast[DType.uint32]()


fn random_f32(inout state: UInt64) -> Float32:
    return (random_u32(state) >> 8).cast[DType.float32]() / RF32_DIV


fn sample_mult(probabilities: UnsafePointer[Scalar[dtype]], n: Int, coin: FLOAT) -> Int:
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

    fn __init__(inout self, filename: StringRef) raises:
        self.vocab_size = 0
        self.token_table = List[String]()
        self.init_ok = 0

        var file: FileHandle

        try:
            file = open(filename, "rb")
        except:
            print("---")
            print("WARNING: Failed to open the tokenizer file", filename)
            print("The Tokenizer is a new feature added April 14 2024.")
            print("Re-run `python train_gpt2.py` to write it")
            print("---")

            self.init_ok = 0
            return

    
        var header = UnsafePointer[Scalar[DType.int32]].alloc(256)
        read_to_dtype_pointer(header,file,256)

        if header[0] != 20240328:
            print("Bad magic model file",header[0])
            exit(1)
        if header[1] != 2:
            print("Bad version in model file", header[1])
            exit(1)

        self.vocab_size = int(header[2])

        for i in range(self.vocab_size):
            var length = int(file.read_bytes(1)[0])
            var str: String = file.read(length)
            if length > 0 and len(str) > 0:
                self.token_table.append(str)
            else:
                self.token_table.append("")

        file.close()
        self.init_ok = 1

    fn decode(self, token_id: Int) -> String:
        if self.init_ok == 0:
            return ""

        if token_id >= 0 and token_id < self.vocab_size:
            return self.token_table[token_id]
        else:
            return ""

    fn safe_printf(self, s: String):
        # the tokens are raw bytes, and we we only want to print the printable ones
        # many bytes can be various control codes, backspace, etc.
        if s == str(NULL):
            return
        if s[0] == "\0":
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

fn read_to_dtype_pointer[T:DType](inout ptr:UnsafePointer[Scalar[T]],file_handle:FileHandle,num:Int,alloc:Bool=False) raises -> None :
    if alloc:
        ptr = UnsafePointer[Scalar[T]].alloc(num)
    _ = file_handle.read(ptr,num)


# ----------------------------------------------------------------------------
# main training loop


fn main() raises:
    # build the GPT-2 model from a checkpoint
    var model = GPT2("gpt2_124M.bin")

    # build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    var tiny_stories_train: StringRef = "./data/TinyStories_train.bin"
    var tiny_stories_val: StringRef = "./data/TinyStories_val.bin"
    var tiny_shakespeare_train: StringRef = "./data/tiny_shakespeare_train.bin"
    var tiny_shakespeare_val: StringRef = "./data/tiny_shakespeare_val.bin"

    var train_tokens: StringRef = tiny_shakespeare_train
    var val_tokens: StringRef = tiny_shakespeare_val

    try:
        var file = open(tiny_shakespeare_train, "r")
        file.close()
    except:
        # both in one go ...
        train_tokens = tiny_stories_train
        val_tokens = tiny_stories_val

    var B: Int = 4
    var T: Int = 64
    var train_loader = DataLoader()
    dataloader_init(train_loader, train_tokens, B, T)
    print("train dataset num_batches:", train_loader.num_batches)
    var val_loader = DataLoader()
    dataloader_init(val_loader, val_tokens, B, T)
    print("val dataset num_batches:", val_loader.num_batches)
    var val_num_batches: Int = 10

    # build the Tokenizer
    var tokenizer = Tokenizer("gpt2_tokenizer.bin")

    # some memory for generating samples from the model
    var rng_state: UInt64 = 1337
    var gen_max_length: Int = 64
    var gen_tokens = UnsafePointer[Scalar[dtype_int]]().alloc(gen_max_length)

    # train

    var elapsed_time_ms_total = 0.0

    for step in range(41):
        # once in a while estimate the validation loss
        if step % 10 == 0:
            var val_loss: FLOAT = 0.0
            dataloader_reset(val_loader)
            for i in range(val_num_batches):
                dataloader_next_batch(val_loader)
                gpt2_forward(model, val_loader.inputs, val_loader.targets, B, T)
                val_loss += model.mean_loss

            val_loss /= val_num_batches
            print("val loss", val_loss)

        # once in a while do model inference to prgenerated INT32 text
        if step > 0 and step % 20 == 0:
            gen_tokens[0] = GPT2_EOT  # the GPT-2 EOT token kicks off the generation

            print("generating:\n---")
            for t in range(1, gen_max_length):
                # note that inference is wasteful here because
                # for each t, we re-compute all activations between 0 and t
                # leaving this alone because you want separate code for inference anyway
                # the inference here is just for sanity checking purposes
                gpt2_forward(model, gen_tokens, NULL_INT, 1, t)
                var probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size
                var coin: FLOAT = random_f32(rng_state).cast[dtype]()
                var next_token: Int = sample_mult(probs, model.config.vocab_size, coin)
                gen_tokens[t] = next_token
                # print the generated token, either using the Tokenizer or a fallback
                if tokenizer.init_ok:
                    var token_str: String = tokenizer.decode(next_token)
                    tokenizer.safe_printf(token_str)

                else:
                    # fall back to printing the token id
                    print(next_token, end=" ")

            print("\n---")

        # do a training step

        var start_time = now()

        dataloader_next_batch(train_loader)
        gpt2_forward(model, train_loader.inputs, train_loader.targets, B, T)
        gpt2_zero_grad(model)
        gpt2_backward(model)
        gpt2_update(model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1)

        var elapsed_time_ms = (now() - start_time) / 1_000_000.0

        elapsed_time_ms_total += elapsed_time_ms

        print(
            "step "
            + str(step)
            + ": train loss "
            + str(model.mean_loss)
            + " (took "
            + str(int(elapsed_time_ms))
            + " ms, average: "
            + str(int(elapsed_time_ms_total / (step + 1)))
            + " ms)"
        )

    # free
    dataloader_free(train_loader)
    dataloader_free(val_loader)
    gpt2_free(model)
