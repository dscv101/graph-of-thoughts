# Language Models

The Language Models module is responsible for managing the large language models (LLMs) used by the Controller.

Currently, the framework supports the following LLMs:
- GPT-4 / GPT-3.5 (Remote - OpenAI API)
- LLaMA-2 (Local - HuggingFace Transformers)
- MCP Language Models (Remote/Local - Model Context Protocol)

The following sections describe how to instantiate individual LLMs and how to add new LLMs to the framework.

## LLM Instantiation
- Create a copy of `config_template.json` named `config.json`.
- Fill configuration details based on the used model (below).

### GPT-4 / GPT-3.5
- Adjust the predefined `chatgpt` or `chatgpt4` configurations or create a new configuration with an unique key.

| Key                 | Value                                                                                                                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_id            | Model name based on [OpenAI model overview](https://platform.openai.com/docs/models/overview).                                                                                                                                                                                                                                                                      |
| prompt_token_cost   | Price per 1000 prompt tokens based on [OpenAI pricing](https://openai.com/pricing), used for calculating cumulative price per LLM instance.                                                                                                                                                                                                                         |
| response_token_cost | Price per 1000 response tokens based on [OpenAI pricing](https://openai.com/pricing), used for calculating cumulative price per LLM instance.                                                                                                                                                                                                                       |
| temperature         | Parameter of OpenAI models that controls the randomness and the creativity of the responses (higher temperature = more diverse and unexpected responses). Value between 0.0 and 2.0, default is 1.0. More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature).     |
| max_tokens          | The maximum number of tokens to generate in the chat completion. Value depends on the maximum context size of the model specified in the [OpenAI model overview](https://platform.openai.com/docs/models/overview). More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens). |
| stop                | String or array of strings specifying sequences of characters which if detected, stops further generation of tokens. More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop).                                                                                                       |
| organization        | Organization to use for the API requests (may be empty).                                                                                                                                                                                                                                                                                                            |
| api_key             | Personal API key that will be used to access OpenAI API.                                                                                                                                                                                                                                                                                                            |

- Instantiate the language model based on the selected configuration key (predefined / custom).
```python
lm = controller.ChatGPT(
    "path/to/config.json", 
    model_name=<configuration key>
)
```

### LLaMA-2
- Requires local hardware to run inference and a HuggingFace account.
- Adjust the predefined `llama7b-hf`, `llama13b-hf` or `llama70b-hf` configurations or create a new configuration with an unique key.

| Key                 | Value                                                                                                                                                                           |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_id            | Specifies HuggingFace LLaMA-2 model identifier (`meta-llama/<model_id>`).                                                                                                       |
| cache_dir           | Local directory where the model will be downloaded and accessed.                                                                                                                    |
| prompt_token_cost   | Price per 1000 prompt tokens (currently not used - local model = no cost).                                                                                                      |
| response_token_cost | Price per 1000 response tokens (currently not used - local model = no cost).                                                                                                    |
| temperature         | Parameter that controls the randomness and the creativity of the responses (higher temperature = more diverse and unexpected responses). Value between 0.0 and 1.0, default is 0.6. |
| top_k               | Top-K sampling method described in [Transformers tutorial](https://huggingface.co/blog/how-to-generate). Default value is set to 10.                                            |
| max_tokens          | The maximum number of tokens to generate in the chat completion. More tokens require more memory.                                                                               |

- Instantiate the language model based on the selected configuration key (predefined / custom).
```python
lm = controller.Llama2HF(
    "path/to/config.json", 
    model_name=<configuration key>
)
```
- Request access to LLaMA-2 via the [Meta form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) using the same email address as for the HuggingFace account.
- After the access is granted, go to [HuggingFace LLaMA-2 model card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), log in and accept the license (a _"You have been granted access to this model"_ message should appear).
- Generate HuggingFace access token.
- Log in from CLI with: `huggingface-cli login --token <your token>`.

### MCP Language Models (Model Context Protocol)
- **Protocol-Compliant Implementation**: Follows the official MCP specification for maximum compatibility
- **Multiple Transport Types**: Supports both stdio and HTTP transports according to MCP standards
- **Advanced Sampling**: Full implementation of MCP sampling protocol with model preferences and context management
- **Validation & Error Handling**: Built-in validation for MCP messages and configurations

#### Configuration
Copy `mcp_config_template.json` to `mcp_config.json` and configure your MCP settings:

**Transport Configuration:**
- `transport.type`: "stdio" for local MCP servers or "http" for remote servers
- `transport.command`: Command to start the MCP server (stdio only)
- `transport.args`: Arguments for the MCP server command (stdio only)
- `transport.url`: URL for HTTP MCP servers (http only)
- `transport.headers`: HTTP headers for authentication (http only)

**Client Information:**
- `client_info.name`: Your application name
- `client_info.version`: Your application version

**Capabilities:**
- `capabilities.sampling`: Enable sampling support

**Default Sampling Parameters:**
- `default_sampling_params.modelPreferences`: Model selection hints and priorities
- `default_sampling_params.temperature`: Sampling temperature (0.0-1.0)
- `default_sampling_params.maxTokens`: Maximum tokens to generate
- `default_sampling_params.includeContext`: Context inclusion ("none", "thisServer", "allServers")

**Connection & Cost Tracking:**
- `connection_config`: Timeout and retry settings
- `cost_tracking`: Token cost configuration for usage tracking

- Instantiate the MCP language model based on the selected configuration key:
```python
lm = language_models.MCPLanguageModel(
    "path/to/mcp_config.json",
    model_name=<configuration key>
)
```

Note: 4-bit quantization is used to reduce the model size for inference. During instantiation, the model is downloaded from HuggingFace into the cache directory specified in the `config.json`. Running queries using larger models will require multiple GPUs (splitting across many GPUs is done automatically by the Transformers library).

## Adding LLMs
More LLMs can be added by following these steps:
- Create a new class as a subclass of `AbstractLanguageModel`.
- Use the constructor for loading the configuration and instantiating the language model (if needed).
```python
class CustomLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        config_path: str = "",
        model_name: str = "llama7b-hf",
        cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        
        # Load data from configuration into variables if needed

        # Instantiate LLM if needed
```
- Implement the `query` abstract method that is used to get a list of responses from the LLM (remote API call or local model inference).
```python
def query(self, query: str, num_responses: int = 1) -> Any:
    # Support caching 
    # Call LLM and retrieve list of responses - based on num_responses    
    # Return LLM response structure (not only raw strings)    
```
- Implement the `get_response_texts` abstract method that is used to get a list of raw texts from the LLM response structure produced by `query`.
```python
def get_response_texts(self, query_response: Union[List[Any], Any]) -> List[str]:
    # Retrieve list of raw strings from the LLM response structure    
```
