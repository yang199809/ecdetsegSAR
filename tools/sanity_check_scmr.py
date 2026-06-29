import torch

from engine.edgecrafter.scmr import SCMR


def main():
    b, n, c_low, q_dim = 2, 5, 192, 192
    coarse_mask = torch.randn(b, n, 32, 32, requires_grad=True)
    low_feat = torch.randn(b, c_low, 64, 64, requires_grad=True)
    scatter_map = torch.randn(b, 1, 16, 16, requires_grad=True).sigmoid()
    query_embed = torch.randn(b, n, q_dim, requires_grad=True)

    scmr = SCMR(c_low=c_low, q_dim=q_dim, hidden_dim=64)
    refined = scmr(coarse_mask, low_feat, scatter_map, query_embed)

    assert refined.shape == (b, n, 64, 64)
    assert refined.requires_grad
    assert torch.isfinite(refined).all()

    loss = refined.mean()
    loss.backward()
    print("SCMR sanity check passed.")


if __name__ == "__main__":
    main()
