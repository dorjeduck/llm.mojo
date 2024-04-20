## How to Use

### Step 1: Install python requirements

Before running the following Python scripts, run this command to install the necessary Python packages:

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

Ensure Mojo is installed by following the [Get started with Mojo 🔥](https://docs.modular.com/mojo/manual/get-started/) guide. Train your model by running:

 ```bash
 mojo train_gpt2.mojo
 ```

This command starts the training process using the prepared data.
