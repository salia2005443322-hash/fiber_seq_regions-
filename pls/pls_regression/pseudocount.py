def lambda_regions(cram, regions, step=75):
    tss_values = []
    for chrom, start, end in regions:
        h1, h2 = window_fetch(cram, chrom, start, end)
        win_len = end - start

        for i in range(0, win_len, step):
            w_end = i + step

            sub_h1 = h1[:, i:w_end] 
            sub_h2 = h2[:, i:w_end] 

            X = np.vstack([sub_h1, sub_h2])

            Xc = X - X.mean(axis=0)
            tss = np.sum(Xc**2)
            tss_values.append(tss)

    return np.median(tss_values)
