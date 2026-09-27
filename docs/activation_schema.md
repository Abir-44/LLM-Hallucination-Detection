# Activation Extraction Handoff Schema

## Extraction decision
The current implemented pipeline uses **Hugging Face Transformers hidden states**, not TransformerLens. Activations are obtained with `output_hidden_states=True` and `outputs.hidden_states`. Only generated-token positions are retained.

## Confirmed Model A configuration
- Model name/version: `Qwen/Qwen2.5-1.5B-Instruct`
- Tokenizer: `Qwen/Qwen2.5-1.5B-Instruct` AutoTokenizer
- Selected hidden-state tuple indices: `[10, 18, 26]`
- Expected hidden dimension: `1536`
- `max_new_tokens`: `80`
- `do_sample`: `False`
- `temperature`: not active / `None`
- `top_p`: not active / `None`
- Random seed: `42` (generation is deterministic with `do_sample=False`; seed is still fixed for reproducibility)
- Device: CUDA if available, otherwise CPU
- Dtype: float16 on CUDA, float32 on CPU

## Model B
**Not yet specified in the repository or handoff request.** The exact model ID, tokenizer, selected layers/indices, hidden dimension, and generation settings must be confirmed before the two-model full run. They should not be guessed.

## Saved file
One PyTorch `.pt` record per prompt:

```text
Prompt_ID                    int
Model_Name                   str
Generated_Response           str
Response_Label               int (0/1) or None
Split                        str
Selected_Hidden_State_Indices list[int]
Hidden_Dimension             int
Hidden_State_Count           int
Generation_Config            dict
Token_Steps                  list[int]
Token_IDs                    list[int]
Token_Texts                  list[str]
Activations                  dict[int, Tensor[token_steps, hidden_dim]]
```

### Mapping to Sakib's requested logical schema
For every token step `t` and selected `Layer_ID = L`:
- `Prompt_ID` = record[`Prompt_ID`]
- `Model_Name` = record[`Model_Name`]
- `Generated_Response` = record[`Generated_Response`]
- `Response_Label` = record[`Response_Label`]
- `Split` = record[`Split`]
- `Token_Step` = record[`Token_Steps`][t]
- `Token_ID` = record[`Token_IDs`][t]
- `Token_Text` = record[`Token_Texts`][t]
- `Layer_ID` = `L`
- `Activation_Vector` = record[`Activations`][L][t]

This nested tensor representation avoids duplicating prompt/model metadata for every token-layer row while preserving the full requested logical schema.

## Loader example
```python
import torch

record = torch.load(
    'data/activations/qwen2_5_1_5b/prompt_1_activations.pt',
    map_location='cpu',
    weights_only=False
)

print(record['Prompt_ID'])
print(record['Model_Name'])
print(record['Generated_Response'])

for layer_id, matrix in record['Activations'].items():
    # matrix shape = [generated_token_count, 1536]
    for token_step in range(matrix.shape[0]):
        token_id = record['Token_IDs'][token_step]
        token_text = record['Token_Texts'][token_step]
        activation_vector = matrix[token_step]
```

## Validation
Run `notebooks/03_activation_extraction.ipynb` from top to bottom. Its 5-prompt validation writes `results/activation_validation_small.json` and checks:
- generated token count
- selected activation shapes
- token/activation time-step alignment
- expected hidden dimension
- NaN/Inf
- Prompt_ID
- model metadata

## Known limitations
1. Only one exact model is currently confirmed; Model B is pending team confirmation.
2. `[10,18,26]` are Hugging Face hidden-state tuple indices. Because `hidden_states[0]` is the embedding output, they should not be casually described as exact transformer block numbers.
3. Existing response labels apply to the previously labeled generated responses. A newly generated response must not automatically inherit a label unless it is the same response under the agreed labeling protocol.
4. This implementation performs generation first and then a forward pass over prompt + exact generated token IDs to extract aligned hidden states. It is not a TransformerLens hook-based online extractor.
