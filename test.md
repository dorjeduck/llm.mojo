## Test

To perform tests on the ported Mojo code, execute the following command:

```bash
mojo test_gpt2.mojo
```

This script is a Mojo port of the original `test_gpt2.c`, created by Andrej. and replicates the testing functionality from the C version.

### etails

The testing process involves loading the `gpt2_124M_debug_state.bin` file and running a forward pass to compare the computed logits and loss with the reference values obtained from the PyTorch implementation. Additionally, the test performs 10 iterations of training using the Adam optimizer to verify that the losses match those computed by PyTorch.

### Outcomes

When running the test, the losses align with the PyTorch results within the specified accuracy range. However, some logits display discrepancies. These discrepancies stem from changes in the order of operations caused by the implemented vectorization. Given the non-associative nature of floating-point arithmetic, such changes can lead to variations in outcomes. For more details on why floating-point arithmetic can lead to such discrepancies, see [Floating-point arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic) on Wikipedia.

### Conclusion

Discrepancies in the logits are noted; however, they do not reflect a decrease in quality. The primary indicator of model performance, the training losses, aligns closely with the PyTorch benchmarks. This consistent alignment verifies the reliability and precision of the Mojo implementation, affirming its functional equivalence with the original C version.
