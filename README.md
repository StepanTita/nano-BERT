# nano-BERT

<img width="1013" alt="transformers-2" src="https://github.com/StepanTita/nano-BERT/assets/44279105/10b80f59-04df-4c34-93d7-889252d0aefb">

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
epochs = 200
batch_size = 32
```

| Dataset                   | Accuracy | F-1 Score |
|---------------------------|----------|-----------|
| IMDB Sentiment (2-class)  | 0.734    | 0.745     |
| HateXplain Data (2-class) | 0.693    | 0.597     |

**Result plots** IMDB:


![accuracy-IMDB](https://github.com/StepanTita/nano-BERT/assets/44279105/a620690d-9561-42a2-9b8d-c257f717caad)
![f1-IMDB](https://github.com/StepanTita/nano-BERT/assets/44279105/e1cdef99-854f-4354-9c5e-7fc9e769f15f)


### Interpretation ⁉️:

#### Attentions Visualized:

![Attention-IMDB-1](https://github.com/StepanTita/nano-BERT/assets/44279105/e25447cd-e6cf-48f9-8b43-870e0f5887e5)
![Attention-IMDB-2](https://github.com/StepanTita/nano-BERT/assets/44279105/a8e5a256-1f48-4278-963d-b1818686436b)
![Attention-IMDB-3](https://github.com/StepanTita/nano-BERT/assets/44279105/c32c7f02-3aae-4850-9bdf-e9b8b679d9c0)
![Attention-IMDB-4](https://github.com/StepanTita/nano-BERT/assets/44279105/45919030-ce55-435a-9e2b-556a42f6c07d)


#### Embeddings Visualized in 3D:

<img width="502" alt="Embeddings-3d-1" src="https://github.com/StepanTita/nano-BERT/assets/44279105/ceb24c65-b210-4a59-b69b-7d1ee98015f7">
<img width="445" alt="Embeddings-3d-2" src="https://github.com/StepanTita/nano-BERT/assets/44279105/49319d39-052d-44d3-9ede-785a06986a1a">
<img width="415" alt="Embeddings-3d-3" src="https://github.com/StepanTita/nano-BERT/assets/44279105/c0d6a62a-7868-49be-86b4-75d75f058fc3">
<img width="347" alt="Embeddings-3d-4" src="https://github.com/StepanTita/nano-BERT/assets/44279105/fa29c671-a95d-4414-9a9e-e2113c677bc4">
<img width="983" alt="Embeddings-3d-5" src="https://github.com/StepanTita/nano-BERT/assets/44279105/158fa412-462e-4eac-a217-dc73c819029a">



_Note: see [demo.ipynb](demo.ipynb) and [imdb_demo.ipynb](imdb_demo.ipynb) for better examples_

## License 📄

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
