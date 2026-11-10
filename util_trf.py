import numpy as np

# Code written by Gemini 3.6 Flash to extract the last layer of the transformer model

def get_trf_vector(doc):
    trf_data = doc._.trf_data

    # 1. Access the underlying array across different spacy-transformers API versions
    if hasattr(trf_data, "last_hidden_layer_state"):
        # Version with DocTransformerOutput attribute
        raw_tensor = trf_data.last_hidden_layer_state
    elif hasattr(trf_data, "tensors") and len(trf_data.tensors) > 0:
        # Version with FullTransformerBatch list
        raw_tensor = trf_data.tensors[0]
    elif hasattr(trf_data, "model_output") and hasattr(
        trf_data.model_output, "last_hidden_state"
    ):
        # HuggingFace raw wrapper
        raw_tensor = trf_data.model_output.last_hidden_state
    else:
        raise AttributeError(
            f"Unable to locate hidden states in {type(trf_data)}"
        )

    # 2. Extract underlying data (handling Thinc / PyTorch / CuPy wrapper)
    if hasattr(raw_tensor, "data"):
        tensor_data = raw_tensor.data
    else:
        tensor_data = raw_tensor

    # 3. Convert tensor to NumPy array
    if hasattr(tensor_data, "get"):
        embeddings = tensor_data.get()  # CuPy GPU array
    elif hasattr(tensor_data, "cpu"):
        embeddings = tensor_data.cpu().numpy()  # PyTorch tensor
    else:
        embeddings = np.asarray(tensor_data)  # Standard NumPy array

    # 4. Mean pooling across token dimensions
    if embeddings.ndim == 3:
        return np.mean(embeddings[0], axis=0)
    
    return np.mean(embeddings, axis=0)