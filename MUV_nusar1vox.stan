data {
  // data dimensions
  int<lower=0> n_voxel;
  int<lower=0> n_frame;
  int<lower=0> n_nus;
  // data index
  array[n_voxel] int<lower=1, upper=n_voxel> idx_voxel;
  array[n_frame] int<lower=1> tr_idx;
  // observations
  array[n_voxel] vector[n_frame] Y;
  // design matrix
  matrix[n_frame, n_nus] X_nus;
  matrix[n_frame, n_voxel] X_hemo;
}

transformed data {
  matrix[n_frame, n_nus] Q_ast;
  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(X_nus) * sqrt(n_frame - 1);
  vector[n_frame - 1] tr_diff = to_vector(tr_idx[2 : n_frame]) 
                              - to_vector(tr_idx[1 : (n_frame - 1)]);
}

parameters {
  // voxel level betas, no hierarchical
  vector[n_voxel] beta_design;
  matrix[n_nus, n_voxel] beta_nus;
  // voxel level scale
  array[n_voxel] real<lower=0> sigma_err;
  // autoregression coeff
  array[n_voxel] real<lower=-1, upper=1> phi;
}

model {
  // priors
  beta_design ~ normal(0, 5);
  to_vector(beta_nus) ~ normal(0, 10);
  sigma_err ~ normal(0, 10);
  phi ~ normal(0, 2);
  // observational model
  vector[n_frame] yrs; // residuals after design and nuisances
  for (n in 1 : n_voxel) {
    yrs = Y[n]
      - X_hemo[ : , n] * beta_design[idx_voxel[n]]
      - Q_ast * beta_nus[ : , idx_voxel[n]];                     
    yrs[2 : n_frame] ~ normal(pow(phi[idx_voxel[n]], tr_diff)
                          .* yrs[1 : (n_frame - 1)],
                          sigma_err[idx_voxel[n]]);
  }
}
