# import torch
# repeter = torch.tensor([2,3])
# a=torch.arange(0,10).view(2,5).repeat_interleave(repeter, dim=0).view(-1)
# print(a)
# # torch.arange(num_agents_batch, device=batch.device).view(batch_size, self.num_agents).repeat_interleave(scatter_add(torch.ones_like(batch), batch), dim=0).view(-1),
import torch

tensor = torch.tensor([[1, 3, 5, 7, 9],
                      [2, 4, 6, 8, 10]])
top_values, top_indices = torch.topk(tensor, 3)

print("Top values:", top_values)       # 输出: [9, 7, 5]
print("Top indices:", top_indices)     # 输出: [4, 3, 2]