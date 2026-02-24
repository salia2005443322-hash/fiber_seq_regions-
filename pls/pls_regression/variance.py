def pls_variance(h1, h2):
    X = np.vstack([h1, h2])[:, :-1]
    Y = np.hstack([h1[:, -1], h2[:, -1]]).reshape(-1, 1)

    pls = PLSRegression(n_components=10, scale=False)
    pls.fit(X, Y)

    Xc = X-pls._x_mean
    Yc = Y-pls._y_mean

    total_var_X = np.sum(Xc**2)
    total_var_Y = np.sum(Yc**2)

    T = pls.x_scores_
    P = pls.x_loadings_
    C = pls.y_loadings_

    ev_X, ev_Y = [], []
    
    for k in range(n_components):
        tk = T[:, [k]]
        pk = P[:, [k]]
        ck = C[:, [k]]

        X_hat_k = tk @ pk.T
        Y_hat_k = tk @ ck.T

        ev_X.append(np.sum(X_hat_k**2) / total_var_X)
        ev_Y.append(np.sum(Y_hat_k**2) / total_var_Y)

    return np.array(ev_X), np.array(ev_Y)
