# llm.🔥

This project is a port of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) to [Mojo](https://docs.modular.com/mojo), currently in beta. It is under active development and subject to changes. Users should expect to encounter bugs and unfinished features.

## Implementation

- **train_gpt2_basic.mojo**: Basic port of train_gpt2.c to Mojo, which does not leverage Mojo's capabilities. Beyond the initial commit, we will not provide further updates for the 'train_gpt2_basic' version, except for necessary bug fixes.
- **train_gpt2.mojo**: Enhanced version utilizing Mojo's performance gems like vectorization and parallelization. Work in progress.

## How to use

Visit [llm.c](https://github.com/karpathy/llm.c) for a detailed explanation of the original project.

Ensure that you have Mojo version 24.2.1 installed. To use llm.mojo, run:

```bash
pip install -r requirements.txt
python prepro_tinyshakespeare.py  
python train_gpt2.py
mojo train_gpt2.mojo
```

For a more detailed step-by-step guide including additional setup details and options, please refer to our [detailed usage instructions](./usage_instructions.md).

## Benchmarks

Basic benchmark results: (M2 MacBook Pro)

- Below are the average training loop times observed across the various programs. Please note that these results are intended to provide a general comparison rather than precise, repeatable metrics.

- We are running the OpenMP-enabled train_gpt2.c with 64 threads.
  (`OMP_NUM_THREADS=64 ./train_gpt2`)

| Implementation             | Average Training Loop Time |
|----------------------------|----------------------------|
| train_gpt2.mojo            | 1819 ms                    |
| train_gpt2.c (with OpenMP) | 1849 ms                    |
| train_gpt2.c (no OpenMP)   | 7473 ms                    |
| train_gpt2_basic.mojo      | 54509 ms                   |

!['Training Loop Times'](imgs/training_loop_times_chart.png)

## Test

We ported `test_gpt2.c` from the original repository to Mojo to validate our port's functionality. For instructions on how to run this test and insights into the results it yields, please see our guide [here](./test.md).

## Development Roadmap

- **Implementation Improvement**: Enhance `train_gpt.mojo` to fully exploit Mojo's capabilities, including further optimization for speed and efficiency.
- **Following Changes of llm.c**: Regularly update the Mojo port to align with the latest improvements and changes made to `llm.c`.
- **Solid Benchmarks**: Develop comprehensive and reliable benchmarks to accurately measure performance improvements and compare them against other implementations.
  
## Changelog

- 2024.04.20
  - Further optimization (utilizing unroll_factor of vectorize)
- 2024.04.19
  - test_gpt2.mojo added.
- 2024.04.18
  - Upgraded project status to Beta.
  - Further optimizations of train_gpt2.mojo.
- 2024.04.16
  - Vectorize parameter update
- 2024.04.15
  - Tokenizer Added - `train_gpt2.c` Update 2024.04.14
  - Bug fix `attention_backward`
- 2024.04.13
  - Initial repository setup and first commit.

## License

MIT
