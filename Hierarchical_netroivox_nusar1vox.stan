data {
  // data dimensions
  int<lower=0> n_roi;
  int<lower=0> n_voxel;
  int<lower=0> n_frame;
  int<lower=0> n_nus;
  // data index
  array[n_voxel] int<lower=1, upper=n_roi> idx_roi;
  array[n_voxel] int<lower=1, upper=n_voxel> idx_voxel;
  array[n_frame] int<lower=1> tr_idx;
  // observations
  array[n_voxel] vector[n_frame] Y;
  // design matrix
  matrix[n_frame, n_nus] X_nus;
  matrix[n_frame, n_voxel] X_hemo;
}

transformed data {
  // thin and scale the QR decomposition
  matrix[n_frame, n_nus] Q_ast;
  Q_ast = qr_thin_Q(X_nus) * sqrt(n_frame - 1);
  vector[n_frame - 1] tr_diff = to_vector(tr_idx[2 : n_frame])
                                - to_vector(tr_idx[1 : (n_frame - 1)]);
}

parameters {
  // network, roi and voxel level betas
  real beta_net;
  vector[n_roi] z_roi;
  vector[n_voxel] z_voxel;
  matrix[n_nus, n_voxel] beta_nus;
  // network, roi and voxel level scales
  array[n_voxel] real<lower=0> sigma_err;
  real<lower=0> sigma_roi;
  real<lower=0> sigma_voxel;
  // autoregression coeff
  array[n_voxel] real<lower=-1, upper=1> phi;
}

model {
  // priors
  beta_net ~ normal(0, 5);
  z_roi ~ std_normal();
  z_voxel ~ std_normal();
  to_vector(beta_nus) ~ normal(0, 10);
  sigma_roi ~ normal(0, 5);
  sigma_voxel ~ normal(0, 5);
  sigma_err ~ normal(0, 10);
  phi ~ normal(0, 2);
  // observational model
  vector[n_frame] yrs; // residuals after design and nuisances
  for (n in 1 : n_voxel) {
    yrs = Y[n]
          - X_hemo[ : , n]
            * (beta_net + sigma_roi .* z_roi[idx_roi[n]]
               + sigma_voxel .* z_voxel[idx_voxel[n]])
          - Q_ast * beta_nus[ : , idx_voxel[n]];
    yrs[2 : n_frame] ~ normal(pow(phi[idx_voxel[n]], tr_diff)
                              .* yrs[1 : (n_frame - 1)],
                              sigma_err[idx_voxel[n]]);    
  }
}
