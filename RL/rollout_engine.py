"""vLLM-backed grouped rollout generation, with a Transformers fallback."""
class RolloutEngine:
    def __init__(self, model_path, tokenizer, *, use_vllm=True, **sampling):
        self.tokenizer, self.sampling = tokenizer, sampling
        self.llm = self.sampling_params = None
        if use_vllm:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_path, dtype="auto")
            self.sampling_params = SamplingParams(**sampling)
        else:
            from transformers import AutoModelForCausalLM
            import torch
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
            self.model.eval()

    def generate(self, prompts, n=4):
        if self.llm is not None:
            from vllm import SamplingParams
            params = SamplingParams(**{**self.sampling, "n": n})
            return [[o.text for o in result.outputs] for result in self.llm.generate(prompts, params)]
        import torch
        groups = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, do_sample=True, num_return_sequences=n,
                                      max_new_tokens=self.sampling.get("max_tokens", 512),
                                      temperature=self.sampling.get("temperature", 0.8),
                                      pad_token_id=self.tokenizer.pad_token_id)
            groups.append([self.tokenizer.decode(x[inputs.input_ids.shape[1]:], skip_special_tokens=True) for x in out])
        return groups
