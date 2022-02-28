import paddle


def test_rel_pos_MHA(device):

    from speechbrain.nnet.attention import RelPosMHAXL

    bsz = 2
    emb_dim = 4
    k_len = [12, 10]
    q_len = [10, 12]
    bias = [True, False]
    head_dim = [4, None]

    for kl in k_len:
        for ql in q_len:
            for b in bias:
                for h in head_dim:
                    relpos = RelPosMHAXL(
                        emb_dim, num_heads=2, vbias=b, vdim=h
                    ).to(device)
                    q = torch.rand((bsz, ql, emb_dim), )
                    k = torch.rand((bsz, kl, emb_dim), )
                    pos_embs = torch.rand(
                        (1, 2 * kl - 1, emb_dim), 
                    )
                    relpos(q, k, k, pos_embs=pos_embs)
