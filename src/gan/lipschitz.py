import torch


def weight_clipping(D, clip_value=0.01):
    for param in D.parameters():
        param.data = torch.clamp(param.data, -clip_value, clip_value)


def gradient_penalty(D, real_samples, fake_samples, device, factor):
    real_samples = real_samples.float()
    fake_samples = fake_samples.float()

    # Get the batch size and shape information
    batch_size = real_samples.size(0)
    # Create alpha with the correct shape for broadcasting

    # Linearly interpolate distribution with:
    # x_interpolated = alpha * x_real + (1-alpha) * x_gen
    alpha = torch.rand((batch_size, 1, 1, 1), device=device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)

    # Take gradient of D's output wrt. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Calculate E[ (L2Norm(grad) - 1)^2 ]
    # L2 Norm can be calculated with Tensor::norm
    # Expected value (E) can be calculated with Tensor::mean
    gradient_norms = gradients.norm(2, dim=1).clamp(min=1e-6)
    gradient_penalty = ((gradient_norms - 1) ** 2).mean()
    return factor * gradient_penalty
