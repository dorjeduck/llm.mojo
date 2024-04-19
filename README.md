# llm.🔥

This project is a port of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) to [Mojo](https://docs.modular.com/mojo), currently in beta. It is under active development and subject to significant changes. Users should expect to encounter bugs and unfinished features.

## Implementation

- **train_gpt_basic.mojo**: Basic port of train_gpt.c to Mojo, which does not leverage Mojo's capabilities. Beyond the initial commit, we will not provide further updates for the 'train_gpt2_basic' version, except for necessary bug fixes.
- **train_gpt.mojo**: Enhanced version utilizing Mojo's performance gems like vectorization and parallelization. Work in progress.

## How to Use

Visit [llm.c](https://github.com/karpathy/llm.c) for a detailed explanation of the original project. To use `llm.mojo`, follow the essential steps below:

### Step 1: Install python requirements

```bash
pip install -r requirements.txt
```

### Step 2: Download and Tokenize a Dataset

Use the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset for a quick setup. This dataset is the fastest to download and tokenize. Run the following command to download and prepare the dataset:

```bash
python prepro_tinyshakespeare.py
```

(all Python scripts in this repo are from Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repository.)

Alternatively, download and tokenize the larger [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset with the following command:

```bash
python prepro_tinystory.py
```

### Step 3: Download the weights

Next download the GPT-2 weights and save them as a checkpoint we can load in Mojo with following command:

```bash
python train_gpt2.py
```

### Step 4: Train the Model

Train your model using the downloaded and tokenized data by running:

 ```bash
 mojo train_gpt.mojo
 ```

## Benchmarks

Preliminary benchmark results: (M2 MacBook Pro)

| Implementation             | Average Training Loop Time |
|----------------------------|----------------------------|
| train_gpt2.mojo             | 1878 ms                    |
| train_gpt2.c (with OpenMP)  | 2119 ms                    |
| train_gpt2.c  (no OpenMP)   | 7473 ms                    |
| train_gpt2_basic.mojo       | 54509 ms                   |

!['Training Loop Times'](imgs/training_loop_times_chart.png)


## Test

To perform tests on the ported Mojo code, execute the following command:

```bash
mojo test_gpt2.mojo
```

This script is a Mojo adaptation of the original `test_gpt2.c`, created by Andrej. and replicates the testing functionality from the C version.

### Test Details

The testing process involves loading the `gpt2_124M_debug_state.bin` file and running a forward pass to compare the computed logits and loss with the reference values obtained from the PyTorch implementation. Additionally, the test performs 10 iterations of training using the Adam optimizer to verify that the losses match those computed by PyTorch.

### Test Outcomes

When running the test, the losses align with the PyTorch results within the specified accuracy range. However, some logits display discrepancies. These discrepancies stem from changes in the order of operations caused by the implemented vectorization. Given the non-associative nature of floating-point arithmetic, such changes can lead to variations in outcomes. For more details on why floating-point arithmetic can lead to such discrepancies, see [Floating-point arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic) on Wikipedia.

### Test Conclusion

Discrepancies in the logits are noted; however, they do not reflect a decrease in quality. The primary indicator of model performance, the training losses, aligns closely with the PyTorch benchmarks. This consistent alignment verifies the reliability and precision of the Mojo implementation, affirming its functional equivalence with the original C version.

## Development Roadmap

- **Implementation Improvement**: Enhance `train_gpt.mojo` to fully exploit Mojo's capabilities, including further optimization for speed and efficiency.
- **Following Changes of llm.c**: Regularly update the Mojo port to align with the latest improvements and changes made to `llm.c`.
- **Solid Benchmarks**: Develop comprehensive and reliable benchmarks to accurately measure performance improvements and compare them against other implementations.
  
## Changelog

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
