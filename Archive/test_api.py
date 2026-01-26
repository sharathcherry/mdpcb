from openai import OpenAI

client = OpenAI(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key='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'
)

print("Testing NVIDIA API...")
try:
    models = client.models.list()
    print("Available models:")
    for m in models.data[:20]:
        if 'llama' in m.id.lower() or 'meta' in m.id.lower():
            print(f"  ★ {m.id}")
        else:
            print(f"    {m.id}")
except Exception as e:
    print(f"Error: {e}")
