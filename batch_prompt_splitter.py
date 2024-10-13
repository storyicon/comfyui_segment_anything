import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Content of current directory: {os.listdir('.')}")
print(f"Content of custom_nodes directory: {os.listdir('../')}")
print(f"Python path: {sys.path}")

class BatchPromptSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "split_char": ("STRING", {"default": ","}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "split_prompts"
    CATEGORY = "text"

    def split_prompts(self, text, split_char):
        prompts = [prompt.strip() for prompt in text.split(split_char)]
        return (prompts,)

NODE_CLASS_MAPPINGS = {
    "BatchPromptSplitter": BatchPromptSplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchPromptSplitter (Philippo)": "Batch Prompt Splitter"
}

print("BatchPromptSplitter node loaded successfully")