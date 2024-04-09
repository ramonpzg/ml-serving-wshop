from mlserver import MLModel
from mlserver.codecs import decode_args
from typing import List

from llama_cpp import Llama

class MyKulModel(MLModel):

    async def load(self):
        self.llm = Llama.from_pretrained(
            repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
            filename="*q8_0.gguf",
            verbose=False
        )

    @decode_args
    async def predict(self, system: List[str], user: List[str]) -> List[str]:

        return [self.llm.create_chat_completion(
            messages = [
                {"role": "system", "content": system[0]},
                {"role": "user", "content": user[0]}
            ]
        )['choices'][0]['message']['content']]
