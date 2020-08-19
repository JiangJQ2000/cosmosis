from builtins import str
from .. import Sampler
import numpy as np


class MaxlikeSampler(Sampler):
    sampler_outputs = [("prior", float), ("like", float), ("post", float)]

    def config(self):
        self.tolerance = self.read_ini("tolerance", float, 1e-3)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.output_ini = self.read_ini("output_ini", str, "")
        self.output_cov = self.read_ini("output_covmat", str, "")
        self.method = self.read_ini("method",str,"Nelder-Mead")
        self.max_posterior = self.read_ini("max_posterior", bool, False)
        self.output_steps = self.read_ini("output_steps", bool, False)
        self.flush_steps = self.read_ini("flush_steps", int, 10)
        self.step_size = self.read_ini("step_size", float, -1.0)
        self.force_file_reopen = self.read_ini("force_file_reopen", bool, False)

        self.ftol = self.read_ini("ftol", float, -1.0)
        self.xtol = self.read_ini("xtol", float, -1.0)
        self.gtol = self.read_ini("gtol", float, -1.0)
        self.grad_eps = self.read_ini("grad_eps", float, -1.0)

        self.lbfgs_maxcor = self.read_ini("lbfgs_maxcor", int, -1)

        if self.max_posterior:
            print("------------------------------------------------")
            print("NOTE: Running optimizer in **max-posterior** mode:")
            print("NOTE: Will maximize the combined likelihood and prior")
            print("------------------------------------------------")
        else:
            print("--------------------------------------------------")
            print("NOTE: Running optimizer in **max-like** mode:")
            print("NOTE: not including the prior, just the likelihood.")
            print("NOTE: Set the parameter max_posterior=T to change this.")
            print("NOTE: This won't matter unless you set some non-flat")
            print("NOTE: priors in a separate priors file.")
            print("--------------------------------------------------")

        self.steps_since_last_flush = 0
        self.converged = False

    def execute(self):
        import scipy.optimize

        def likefn(p_in):
            #Check the normalization
            if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
                return np.inf
            p = self.pipeline.denormalize_vector(p_in)
            
            if self.output_steps:
                results = self.pipeline.run_results(p)
                self.output.parameters(p, results.extra, results.prior, results.like, results.post)
                if self.steps_since_last_flush > self.flush_steps:
                    self.output.flush()
                    self.steps_since_last_flush = 0
                    if self.force_file_reopen:
                        self.output._file.close()
                        self.output._file = open(self.output._filename, "a+")
                else:
                    self.steps_since_last_flush += 1
                if self.max_posterior:
                    return -results.post
                else:
                    return -results.like
            else:
                if self.max_posterior:
                    like, extra = self.pipeline.posterior(p)
                    self.output.log_debug("%s  post=%f"%('   '.join(str(x) for x in p),like))
                else:
                    like, extra = self.pipeline.likelihood(p)
                    self.output.log_debug("%s  like=%f"%('   '.join(str(x) for x in p),like))
                return -like

        # starting position in the normalized space.  This will be taken from
        # a previous sampler if available, or the values file if not.
        start_vector = self.pipeline.normalize_vector(self.start_estimate())
        bounds = [(0.0, 1.0) for p in self.pipeline.varied_params]

        options = {'maxiter':self.maxiter, 'disp':True}
        if self.method == "Nelder-Mead":
            print("Using adaptive Nelder-Mead")
            options = {**options, 'xatol':1e-3, 'fatol':1e-2, 'adaptive':True}
        elif self.method == "Powell":
            print(f"Using {self.method}")
            if self.step_size > 0:
                print(f"Using step size {self.step_size}")
                search_directions = self.step_size * np.eye(len(self.pipeline.varied_params))
                options = {**options, "direc" : search_directions}
        elif self.method == "L-BFGS-B":
            if self.ftol > 0.0:
                options["ftol"] = self.ftol
            if self.gtol > 0.0:
                options["gtol"] = self.gtol
            if self.grad_eps > 0.0:
                options["eps"] = self.grad_eps
            if self.lbfgs_maxcor > 0:
                options["maxcor"] = self.lbfgs_maxcor
            else:
                options["maxcor"] = len(self.pipeline.varied_params)*2
            print(f"Using {self.method} with options {options}")
        else:
            print(f"Using {self.method}")

        result = scipy.optimize.minimize(likefn, start_vector, method=self.method, 
          tol=self.tolerance,  bounds=bounds, 
          options=options)

        opt_norm = result.x
        opt = self.pipeline.denormalize_vector(opt_norm)
        

        #Some output - first log the parameters to the screen.
        #It's not really a warning - that's just a level name
        results = self.pipeline.run_results(opt)
        if self.max_posterior:

            self.output.log_warning("Best fit (by posterior):\n%s"%'   '.join(str(x) for x in opt))
        else:
            self.output.log_warning("Best fit (by likelihood):\n%s"%'   '.join(str(x) for x in opt))
        self.output.log_warning("Posterior: {}\n".format(results.post))
        self.output.log_warning("Likelihood: {}\n".format(results.like))

        #Next save them to the proper table file
        self.output.parameters(opt, results.extra, results.prior, results.like, results.post)

        #If requested, create a new ini file for the
        #best fit.
        if self.output_ini:
          self.pipeline.create_ini(opt, self.output_ini)

        self.distribution_hints.set_peak(opt)          

        #Also if requested, approximate the covariance matrix with the 
        #inverse of the Hessian matrix.
        #For a gaussian likelihood this is exact.
        covmat = None
        if hasattr(result, 'hess_inv'):
            if self.method == "L-BFGS-B":
                covmat = self.pipeline.denormalize_matrix(result.hess_inv.todense())
            else:
                covmat = self.pipeline.denormalize_matrix(result.hess_inv)
        elif hasattr(result, 'hess'):
            covmat = self.pipeline.denormalize_matrix(np.linalg.inv(result.hess_inv))

        if covmat is None:
            if self.output_cov:
               self.output.log_error("Sorry - the optimization method you chose does not return a covariance (or Hessian) matrix")
        else:
            if self.output_cov:
                np.savetxt(self.output_cov, covmat)
            self.distribution_hints.set_cov(covmat)

        self.converged = True

    def is_converged(self):
        return self.converged
