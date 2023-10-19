# nano-BERT

![Attention is all you need transformer architecture](link/to/logo.png)

### Nano-BERT: A Simplified and Understandable Implementation of BERT

**Nano-BERT** is a straightforward, lightweight and comprehensible custom implementation of BERT, inspired by the
foundational "Attention is All You Need" paper. The primary objective of this project is to distill the essence of
transformers by simplifying the complexities and unnecessary details, making it an ideal starting point for those aiming
to grasp the fundamental ideas behind transformers.

Key Features and Focus 🚀:

- **Simplicity and Understandability**: Nano-BERT prioritizes simplicity and clarity, making it accessible for anyone
  looking to understand the core concepts of transformers.


- **Multi-Headed Self Attention**: The implementation of multi-headed self-attention is intentionally less efficient but
  more descriptive. Each attention head is treated as a separate object, emphasizing transparency over optimization
  techniques like matrix transposition and efficient multiplication.


- **Educational Purposes**: This project is designed for educational purposes, offering a learning platform for
  individuals interested in transformer architectures.


- **Customizability**: Nano-BERT allows extensive customization, enabling users to experiment with various parameters
  such as the number of layers, heads, and embedding sizes. It serves as a playground for exploring the impact of
  different configurations on model performance.


- **Inspiration**: The project draws inspiration from ongoing research endeavors related to efficient LLM
  fine-tuning [space-model](https://github.com/StepanTita/space-model). Additionally, it is influenced by the deep
  learning series conducted by Andrej Karpathy [YouTube](https://www.youtube.com/@AndrejKarpathy), particularly
  the [nanoGPT](https://github.com/karpathy/nanoGPT) project.


- **Motivation and Development**:
  Nano-BERT originated from the author's curiosity about embedding custom datasets into a three-dimensional space using
  BERT. To achieve this, the goal was to construct a fully customizable version of BERT, providing complete control over
  the model's behavior. The motivation was to comprehend how BERT could handle datasets with words as tokens, diverging
  from the common sub-word approach.

**Community Engagement 💬**:
While Nano-BERT is not intended for production use, contributions, suggestions, and feedback from the community are
highly encouraged. Users are welcome to propose improvements, simplifications, or enhanced descriptions by creating pull
requests or issues.

**Exploration and Experimentation 🌎**:
Nano-BERT's flexibility enables users to experiment freely. Parameters like the number of layers, heads, and embedding
sizes can be tailored to specific needs. This customizable nature empowers users to explore diverse configurations and
assess their impact on model outcomes.

_Note: Nano-BERT was developed with a focus on educational exploration and understanding, and it should be utilized
within the scope of educational and experimental contexts only!_

## Installation 🛠️

### Prerequisites

- Python 3.10.x
- pip*

```
pip install torch
```

_Note: to be able to run demos you might need some additional packages, but for base model all you needs is pytorch_

```
pip install tqdm scikit-learn matplotlib plotly
```

### Package installation

⚠️: currently only available through GitHub, but `pip` version is coming soon!

```bash
git clone https://github.com/StepanTita/nano-BERT.git
```

## Usage Example ⚙️

```python
from nano_bert.model import NanoBERT
from nano_bert.tokenizer import WordTokenizer

vocab = [...]  # a list of tokens (or words) to use in tokenizer

tokenizer = WordTokenizer(vocab=vocab, max_seq_len=128)

# Usage:
input_ids = tokenizer('This is a sentence')  # or tokenizer(['This', 'is', 'a', 'sentence'])

# Instantiate the NanoBERT model
nano_bert = NanoBERT(input_ids)

# Example usage
embedded_text = nano_bert.embedding(input_ids)
print(embedded_text)
```

## Results 📈:

### Benchmarks 🏆:

For all of the following experiments we use the following configuration:

```
n_layer = 1
n_head = 1
dropout = 0.1
n_embed = 3
max_seq_len = 128
```

| Dataset                   | Accuracy | F-1 Score |
|---------------------------|----------|-----------|
| IMDB Sentiment (2-class)  | TBD      | TBD       |
| HateXplain Data (2-class) | TBD      | TBD       |

### Interpretation ⁉️:

#### Attentions Visualized:
...

#### Embeddings Visualized in 3D:
...

_Note: see [demo.ipynb](demo.ipynb) and [imdb_demo.ipynb](imdb_demo.ipynb) for better examples_

## License 📄

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.