# 前向传播生成器G_AB和G_BA
fake_B = G_AB(real_A)
fake_A = G_BA(real_B)
# 前向传播新的判别器D_new
pred_A, _ = D_new(fake_A)
pred_B, _ = D_new(fake_B)
# 计算生成器的损失
loss_G_AB = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B))) + criterion_GAN(D_new(fake_B), torch.zeros_like(D_new(fake_B))) + lambda_cycle * criterion_cycle(real_A, G_BA(fake_B)) + lambda_identity * criterion_identity(real_B, G_BA(real_B))
loss_G_BA = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A))) + criterion_GAN(D_new(fake_A), torch.ones_like(D_new(fake_A))) + lambda_cycle * criterion_cycle(real_B, G_AB(fake_A)) + lambda_identity * criterion_identity(real_A, G_AB(real_A))
# 反向传播和优化生成器
optimizer_G.zero_grad()
loss_G_AB.backward()
loss_G_BA.backward()
optimizer_G.step()
