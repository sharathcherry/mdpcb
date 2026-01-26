from openai import OpenAI

# Initialize NVIDIA API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"  # replace with your NVIDIA key
)

# Create a streaming chat completion
completion = client.chat.completions.create(
    model="writer/palmyra-med-70b-32k",
    messages=[{"role": "user", "content": "Write a short paragraph about AI in healthcare"}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=512,
    stream=True
)

# Print streaming output
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
