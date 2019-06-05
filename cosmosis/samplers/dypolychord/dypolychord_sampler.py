#coding: utf-8
from __future__ import print_function
from builtins import str
from builtins import range
from .. import ParallelSampler
import ctypes as ct
import os
import cosmosis
import numpy as np
import sys

DYPOLYCHORD_SECTION='dypolychord'


class DyPolyChordSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("post", float), ("weight", float)]
    supports_smp=False
    understands_fast_subspaces = True

    def config(self):
        try:
            import dyPolyChord.pypolychord_utils
            import dyPolyChord
        except ImportError:
            raise

        if self.pool:
            self.MPI_comm = self.pool.comm
        else:
            self.MPI_comm = None

        self.converged=False

        self.ndim = len(self.pipeline.varied_params)
        self.nderived = len(self.pipeline.extra_saves)

        self.do_dynamic_nested_sampling = self.read_ini("do_dynamic_nested_sampling", bool, True)

        #dyPolyChord options
        self.dynamic_goal          = self.read_ini("dynamic_goal", float, 1.0)
        self.nlive_const           = self.read_ini("nlive_const", int, 100)
        self.ninit                 = self.read_ini("ninit", int, 10)

        #Output and feedback options
        self.feedback               = self.read_ini("feedback", int, 1)
        self.file_root              = self.read_ini("file_root", str, "")
        self.base_dir               = self.read_ini("base_dir", str, ".")
        self.compression_factor     = self.read_ini("compression_factor", float, np.exp(-1))

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "clusters"), exist_ok=True)

        self.max_iterations              = self.read_ini("max_iterations", int, -1)
        self.precision_criterion         = self.read_ini("precision_criterion", float, 0.01)
        self.fast_fraction    = self.read_ini("fast_fraction", float, 0.5)

        if self.pipeline.do_fast_slow:
            self.grade_dims = self.pipeline.n_slow_params, self.pipeline.n_fast_params
            self.grade_frac = 1 - self.fast_fraction, self.fast_fraction
            print("Telling Polychord to spend fraction {} if its time in the fast subspace (adjust with fast_fraction option)".format(self.fast_fraction))
        else:
            self.grade_dims = [self.pipeline.nvaried]
            self.grade_frac = [1.0]

        self.polychord_settings_dict = {"feedback"            : self.feedback,
                                        "precision_criterion" : self.precision_criterion,
                                        "max_ndead"           : self.max_iterations,
                                        "compression_factor"  : self.compression_factor,
                                        "grade_dims"          : self.grade_dims,
                                        "grade_frac"          : self.grade_frac,
                                        "base_dir"            : self.base_dir,
                                        "file_root"           : self.file_root,
                                        }
        try:
            random_seed                 = self.read_ini("random_seed", int)
            self.polychord_settings_dict["seed"] = random_seed
        except:
            pass

        if not self.do_dynamic_nested_sampling:
            #Polychord options
            self.num_repeats                 = self.read_ini("num_repeats", int, 0)
            if self.num_repeats == 0:
                self.num_repeats = 5 * self.grade_dims[0]
                print("Polychord num_repeats = {}  (5 * n_slow_params [{}])".format(self.num_repeats, self.grade_dims[0]))
            else:
                print("Polychord num_repeats = {}  (from parameter file)".format(self.num_repeats))

            self.live_points            = self.read_ini("live_points", int, 100)
            self.resume                 = self.read_ini("resume", bool, False)
            
            self.nprior                      = self.read_ini("nprior", int, -1)
            self.log_zero                    = self.read_ini("log_zero", float, -1e30)
            self.weighted_posteriors         = self.read_ini("weighted_posteriors", bool, True)
            self.equally_weighted_posteriors = self.read_ini("equally_weighted_posteriors", bool, False)       
            self.do_clustering               = self.read_ini("do_clustering", bool, True)
            
            self.polychord_settings_dict.update({"nlive"               : self.live_points,
                                                 "num_repeats"         : self.num_repeats,
                                                 "nprior"              : self.nprior,
                                                 "do_clustering"       : self.do_clustering,
                                                 "posteriors"          : self.weighted_posteriors,
                                                 "equals"              : self.equally_weighted_posteriors,
                                                 "write_dead"          : True,
                                                 "write_stats"         : True,
                                                 "write_paramnames"    : False,
                                                 "write_prior"         : False,
                                                 "write_live"          : False,
                                                 "write_resume"        : False,
                                                 "read_resume"         : False,
                                                })


        if self.output:
            def dumper(live, dead, logweights, log_z, log_z_err):
                print("Saving %d samples" % len(dead))
                self.output_params(live, dead, logweights, log_z, log_z_err)
        else:
            def dumper(live, dead, logweights, log_z, log_z_err):
                pass

        def prior(cube):
            if self.pipeline.do_fast_slow:
                cube = self.reorder_slow_fast(cube)
            try:
                theta = self.pipeline.denormalize_vector_from_prior(cube) 
            except ValueError:
                # Polychord sometimes seems to propose outside the prior.
                # Just give terrible parameters when that happens.
                theta = np.repeat(-np.inf, self.ndim)
            return theta
  
        def loglikelihood(theta):
            if np.any(~np.isfinite(theta)):
                logL = -np.inf
                phi = np.zeros(self.nderived)
            try:
                logL, phi = self.pipeline.likelihood(theta)
            except KeyboardInterrupt:
                raise sys.exit(1)

            return logL, phi

        self.polychord_callable = dyPolyChord.pypolychord_utils.RunPyPolyChord(
                        likelihood=loglikelihood, prior=prior, 
                        ndim=self.ndim, nderived=self.nderived, dumper=dumper)

        

    def reorder_slow_fast(self, x):
        y = np.zeros_like(x)
        ns = self.pipeline.n_slow_params
        y[self.pipeline.slow_param_indices] = x[0:ns]
        y[self.pipeline.fast_param_indices] = x[ns:]
        return y

    def worker(self):
        self.sample()

    def execute(self):
        self.log_z = 0.0
        self.log_z_err = 0.0

        self.sample()

        self.output.final("log_z", self.log_z)
        self.output.final("log_z_error", self.log_z_err)

    def sample(self):
        if self.do_dynamic_nested_sampling:
            import dyPolyChord
            # Run dyPolyChord
            dyPolyChord.run_dypolychord(run_polychord=self.polychord_callable, 
                                        dynamic_goal=self.dynamic_goal, 
                                        settings_dict_in=self.polychord_settings_dict,
                                        nlive_const=self.nlive_const,
                                        ninit=self.ninit,
                                        comm=self.MPI_comm)
        else:
            # Run PolyChord
            self.polychord_callable(settings_dict=self.polychord_settings_dict, 
                                    comm=self.MPI_comm)

        self.converged = True

    def output_params(self, live, dead, logweights, log_z, log_z_err):
        self.log_z = log_z
        self.log_z_err = log_z_err
        ndead = dead.shape[0]
        for i in range(ndead):
            params = dead[i,:self.ndim]
            extra_vals = dead[i,self.ndim:self.ndim+self.nderived]
            birth_like = dead[i,self.ndim+self.nderived]
            like = dead[i,self.ndim+self.nderived+1]
            importance = np.exp(logweights[i])
            self.output.parameters(params, extra_vals, like, importance)
        self.output.final("nsample", ndead)
        self.output.flush()

    def is_converged(self):
        return self.converged
