data {
  // data dimensions
  int<lower=0> n_voxel;
  int<lower=0> n_frame;
  int<lower=0> n_nus;
  int<lower=0> n_hemo;
  // data index
  array[n_frame] int<lower=1> tr_idx;
  array[n_hemo] int<lower=1> hemo_idx;
  // observations
  array[n_voxel] vector[n_frame] Y;
  // design matrix
  vector[n_frame] X;
  // nuisance matrix
  matrix[n_frame, n_nus] X_nus;
}

transformed data {
  // thin and scale the QR decomposition
  matrix[n_frame, n_nus] Q_ast;
  Q_ast = qr_thin_Q(X_nus) * sqrt(n_frame - 1);
  matrix[n_frame, 1 + n_nus] X_all = append_col(X, Q_ast);
  vector[n_frame - 1] tr_diff = to_vector(tr_idx[2 : n_frame])
                                - to_vector(tr_idx[1 : (n_frame - 1)]);
}

parameters {
  // network, roi and voxel level betas
  real beta_roi;
  vector[n_voxel] z_voxel;
  matrix[n_nus, n_voxel] beta_nus;
  // network, roi and voxel level scales
  array[n_voxel] real<lower=0> sigma_err;
  real<lower=0> sigma_voxel;
  // autoregression coeff
  array[n_voxel] real<lower=-1, upper=1> phi;
}

model {
  // priors
  beta_roi ~ normal(0, 5);
  z_voxel ~ std_normal();
  to_vector(beta_nus) ~ normal(0, 10);
  sigma_voxel ~ normal(0, 5);
  sigma_err ~ normal(0, 10);
  phi ~ normal(0, 2);
  // observational model
  vector[n_frame] yrs; // residuals after design and nuisances
  for (n in 1 : n_voxel) {
    yrs = Y[n]
          - X_all
            * append_row(beta_roi + sigma_voxel .* z_voxel[n],
                         beta_nus[ : , n]);
    yrs[2 : n_frame] ~ normal(pow(phi[n], tr_diff)
                              .* yrs[1 : (n_frame - 1)],
                              sigma_err[n]);    
  }
}

generated quantities {
  vector[n_voxel * n_hemo] log_lik;
  vector[n_voxel] r_squared;
  // "local" environment so temp doesn't get saved to file
  {
    matrix[n_voxel, n_hemo] temp;
    for (n in 1 : n_voxel) {
      for (t in 1 : n_hemo) {
        temp[n, t] = normal_lpdf(Y[n, hemo_idx[t]] | X_all[hemo_idx[t],  : ]
                                                     * append_row(beta_roi
                                                                  + sigma_voxel
                                                                    .* z_voxel[n],
                                                                  beta_nus[ : , n])
                                                     + pow(phi[n],
                                                           tr_idx[hemo_idx[t]]
                                                           - tr_idx[hemo_idx[t] - 1])
                                                       * (Y[n, (hemo_idx[t] - 1)]
                                                          - (X_all[(hemo_idx[t] - 1),  : ]
                                                             * append_row(
                                                             beta_roi
                                                             + sigma_voxel
                                                               .* z_voxel[n],
                                                             beta_nus[ : , n]))),
                                                     sigma_err[n]);
      }
    }
    log_lik = to_vector(temp);
    
    matrix[n_voxel, n_frame] ypred;
    for (n in 1 : n_voxel) {
      ypred[n, 1] = X_all[1,  : ]
                    * append_row(beta_roi + sigma_voxel .* z_voxel[n],
                                 beta_nus[ : , n])
                    + sigma_err[n];
      for (t in 2 : n_frame) {
        ypred[n, t] = X_all[t,  : ]
                      * append_row(beta_roi + sigma_voxel .* z_voxel[n],
                                   beta_nus[ : , n])
                      + pow(phi[n], tr_idx[t] - tr_idx[t - 1])
                        * (ypred[n, (t - 1)]
                           - (X_all[(t - 1),  : ]
                              * append_row(beta_roi + sigma_voxel .* z_voxel[n],
                                           beta_nus[ : , n])))
                      + sigma_err[n];
      }
      r_squared[n] = variance(ypred[n , hemo_idx]) / (variance(ypred[n , hemo_idx]) 
                                                      + pow(sigma_err[n], 2));
    }
  }
}
