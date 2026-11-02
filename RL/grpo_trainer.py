"""One optimizer update over a previously collected GRPO rollout buffer."""
import torch
import torch.nn.functional as F

def _sequence_logps(model, ids, mask, prompt_lengths):
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    labels = ids[:, 1:]
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, ids.size(1), device=ids.device).unsqueeze(0)
    response_mask = (positions >= prompt_lengths.unsqueeze(1)) & mask[:, 1:].bool()
    return (token_logps * response_mask).sum(-1) / response_mask.sum(-1).clamp_min(1)

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, optimizer, *, clip_eps=0.2, beta=0.04):
        self.model, self.ref_model, self.tokenizer, self.optimizer = model, ref_model, tokenizer, optimizer
        self.clip_eps, self.beta = clip_eps, beta

    def update(self, buffer):
        device = next(self.model.parameters()).device
        ids, mask = buffer["input_ids"].to(device), buffer["attention_mask"].to(device)
        prompt_lens = buffer["prompt_lens"].to(device)
        old, ref = buffer["old_logps"].to(device), buffer["ref_logps"].to(device)
        advantages = buffer["advantages"].to(device)
        new = _sequence_logps(self.model, ids, mask, prompt_lens)
        ratio = torch.exp(new - old)
        policy_loss = -torch.minimum(ratio * advantages,
            ratio.clamp(1-self.clip_eps, 1+self.clip_eps) * advantages).mean()
        loss = policy_loss + self.beta * (new-ref).mean()
        self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step()
        return {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                "mean_reward": float(buffer["rewards"].mean()),
                "accuracy": float((buffer["rewards"] > 0.1).float().mean())}

