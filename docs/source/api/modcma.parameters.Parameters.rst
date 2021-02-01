Parameters
==========

.. currentmodule:: modcma.parameters

.. autoclass:: Parameters
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Parameters.a_tpa
      ~Parameters.active
      ~Parameters.b_tpa
      ~Parameters.base_sampler
      ~Parameters.bound_correction
      ~Parameters.budget
      ~Parameters.c1
      ~Parameters.cc
      ~Parameters.cmu
      ~Parameters.compute_termination_criteria
      ~Parameters.condition_cov
      ~Parameters.cs
      ~Parameters.d
      ~Parameters.decay_factor
      ~Parameters.elitist
      ~Parameters.init_sigma
      ~Parameters.init_threshold
      ~Parameters.ipop_factor
      ~Parameters.lambda_
      ~Parameters.last_restart
      ~Parameters.lb
      ~Parameters.local_restart
      ~Parameters.max_resamples
      ~Parameters.mirrored
      ~Parameters.mu
      ~Parameters.old_population
      ~Parameters.orthogonal
      ~Parameters.population
      ~Parameters.ps_factor
      ~Parameters.seq_cutoff_factor
      ~Parameters.sequential
      ~Parameters.step_size_adaptation
      ~Parameters.target
      ~Parameters.termination_criteria
      ~Parameters.threshold
      ~Parameters.threshold_convergence
      ~Parameters.tolup_sigma
      ~Parameters.tolx
      ~Parameters.ub
      ~Parameters.weights_option

   .. rubric:: Methods Summary

   .. autosummary::

      ~Parameters.adapt
      ~Parameters.adapt_covariance_matrix
      ~Parameters.adapt_sigma
      ~Parameters.calculate_termination_criteria
      ~Parameters.from_config_array
      ~Parameters.get_sampler
      ~Parameters.init_adaptation_parameters
      ~Parameters.init_dynamic_parameters
      ~Parameters.init_fixed_parameters
      ~Parameters.init_local_restart_parameters
      ~Parameters.init_selection_parameters
      ~Parameters.load
      ~Parameters.perform_eigendecomposition
      ~Parameters.perform_local_restart
      ~Parameters.record_statistics
      ~Parameters.save

   .. rubric:: Attributes Documentation

   .. autoattribute:: a_tpa
   .. autoattribute:: active
   .. autoattribute:: b_tpa
   .. autoattribute:: base_sampler
   .. autoattribute:: bound_correction
   .. autoattribute:: budget
   .. autoattribute:: c1
   .. autoattribute:: cc
   .. autoattribute:: cmu
   .. autoattribute:: compute_termination_criteria
   .. autoattribute:: condition_cov
   .. autoattribute:: cs
   .. autoattribute:: d
   .. autoattribute:: decay_factor
   .. autoattribute:: elitist
   .. autoattribute:: init_sigma
   .. autoattribute:: init_threshold
   .. autoattribute:: ipop_factor
   .. autoattribute:: lambda_
   .. autoattribute:: last_restart
   .. autoattribute:: lb
   .. autoattribute:: local_restart
   .. autoattribute:: max_resamples
   .. autoattribute:: mirrored
   .. autoattribute:: mu
   .. autoattribute:: old_population
   .. autoattribute:: orthogonal
   .. autoattribute:: population
   .. autoattribute:: ps_factor
   .. autoattribute:: seq_cutoff_factor
   .. autoattribute:: sequential
   .. autoattribute:: step_size_adaptation
   .. autoattribute:: target
   .. autoattribute:: termination_criteria
   .. autoattribute:: threshold
   .. autoattribute:: threshold_convergence
   .. autoattribute:: tolup_sigma
   .. autoattribute:: tolx
   .. autoattribute:: ub
   .. autoattribute:: weights_option

   .. rubric:: Methods Documentation

   .. automethod:: adapt
   .. automethod:: adapt_covariance_matrix
   .. automethod:: adapt_sigma
   .. automethod:: calculate_termination_criteria
   .. automethod:: from_config_array
   .. automethod:: get_sampler
   .. automethod:: init_adaptation_parameters
   .. automethod:: init_dynamic_parameters
   .. automethod:: init_fixed_parameters
   .. automethod:: init_local_restart_parameters
   .. automethod:: init_selection_parameters
   .. automethod:: load
   .. automethod:: perform_eigendecomposition
   .. automethod:: perform_local_restart
   .. automethod:: record_statistics
   .. automethod:: save
