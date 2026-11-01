"""Minimal clipped GRPO update with rule-verifier rewards and reference KL."""
import torch
import torch.nn.functional as F
from verifier.math_evaluator import rule_reward

def _sequence_logps(model, ids, mask, prompt_lengths):
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    labels = ids[:, 1:]
    logps = F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, ids.size(1), device=ids.device).unsqueeze(0)
    response_mask = (positions >= prompt_lengths.unsqueeze(1)) & mask[:, 1:].bool()
    return (logps * response_mask).sum(-1) / response_mask.sum(-1).clamp_min(1)

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, optimizer, *, clip_eps=0.2, beta=0.04, n_generations=4):
        self.model, self.ref_model, self.tokenizer, self.optimizer = model, ref_model, tokenizer, optimizer
        self.clip_eps, self.beta, self.n_generations = clip_eps, beta, n_generations

    def step(self, prompts, references, rollout_engine):
        groups = rollout_engine.generate(prompts, self.n_generations)
        flat_p = [p for p, group in zip(prompts, groups) for _ in group]
        completions = [c for group in groups for c in group]
        refs = [r for r, group in zip(references, groups) for _ in group]
        rewards = torch.tensor([rule_reward(c, r)[0] for c, r in zip(completions, refs)],
                               device=next(self.model.parameters()).device)
        shaped = rewards.view(len(prompts), -1)
        advantages = ((shaped - shaped.mean(-1, keepdim=True)) /
                      (shaped.std(-1, keepdim=True, unbiased=False) + 1e-4)).flatten()
        encoded = self.tokenizer([p+c for p,c in zip(flat_p, completions)], padding=True,
                                 truncation=True, return_tensors="pt").to(rewards.device)
        p_tokens = self.tokenizer(flat_p, padding=True, truncation=True, return_tensors="pt")
        prompt_lens = p_tokens.attention_mask.sum(-1).to(rewards.device)
        with torch.no_grad():
            old = _sequence_logps(self.model, encoded.input_ids, encoded.attention_mask, prompt_lens)
            ref = _sequence_logps(self.ref_model, encoded.input_ids, encoded.attention_mask, prompt_lens)
        new = _sequence_logps(self.model, encoded.input_ids, encoded.attention_mask, prompt_lens)
        ratio = torch.exp(new-old)
        loss = -torch.minimum(ratio*advantages, ratio.clamp(1-self.clip_eps,1+self.clip_eps)*advantages).mean()
        loss += self.beta * (new-ref).mean()
        self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step()
        return {"loss": float(loss.detach()), "mean_reward": float(rewards.mean()),
                "accuracy": float((rewards > 0.1).float().mean())}
