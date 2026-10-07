"""rings2cosmo: stellar-dynamics + strong-lensing fits of the PPN parameter gamma.

Model: Schwab, Bolton & Rappaport (2010), arXiv:0907.4992v2.

The module runs in two modes, chosen by the arguments you pass.

Constant gamma (the original behaviour; 4 parameters: alpha, beta, delta, gamma)
    sampler = logprobability_sampling(z_S, z_L, velDisp, velDispErr, theta_E,
                                      seeing_atm, theta_ap, gamma_ini=1.0)

Varied gamma (5 parameters: alpha, beta, delta, gamma_0, gamma_1)
    gamma_i = gamma_0 + gamma_1 * x_i, where x_i is a normalised property of lens i.
    Switched on by giving a starting value for gamma_1 and naming the property:

    sampler = logprobability_sampling(z_S, z_L, velDisp, velDispErr, theta_E,
                                      seeing_atm, theta_ap, gamma_ini=1.0,
                                      gamma_1_ini=0.0, variable='z_L')

    `variable` is one of GAMMA_VARIABLES; see gamma_covariate() for how each is
    normalised. `gamma_0_ini` is accepted as another name for `gamma_ini`.

All angles are in radians and velocity dispersions in km/s.
"""
import warnings

import emcee
import numpy as np
import scipy as sp
import math

from scipy.optimize import minimize
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from multiprocessing import Pool

# Cosmology used:
# From Wikipedia (https://en.wikipedia.org/wiki/Lambda-CDM_model)
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

# Physical constants:
c = (const.c).to(u.km/u.second)
clight = c.value

# Lens properties gamma may depend on in a varied run.
GAMMA_VARIABLES = ('z_S', 'z_L', 'theta_E', 'velDisp', 'DLS', 'DL', 'DS')


def ratio_gamma(x):
    """Eq. (15) from arXiv:0907.4992v2

    Args:
        x (float): parameter

    Returns:
        float: ratio of gamma funcions.
    """
    return math.gamma((x - 1) / 2) / math.gamma(x / 2)


def _centred(x):
    """(x - mean) / (max - min): zero for the average lens, spans a range of 1."""
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.max(x) - np.min(x))


def _covariate(variable, z_S, z_L, theta_E, velDisp, DS=None, DL=None, DLS=None):
    """Normalised lens property for a varied run (distances are plain floats in Mpc)."""
    if variable not in GAMMA_VARIABLES:
        raise ValueError(f"variable must be one of {GAMMA_VARIABLES}, got {variable!r}")
    if variable == 'z_S':
        return _centred(1 + np.asarray(z_S, dtype=float))
    if variable == 'z_L':
        return _centred(1 + np.asarray(z_L, dtype=float))
    if variable == 'theta_E':
        return _centred((np.asarray(theta_E, dtype=float) * u.rad).to(u.arcsec).value)
    if variable == 'velDisp':
        if velDisp is None:
            raise ValueError("variable='velDisp' needs the measured velocity dispersions (velDisp)")
        return _centred(velDisp)
    if variable == 'DLS':
        D = cosmo.angular_diameter_distance_z1z2(z_L, z_S).value if DLS is None else DLS
    elif variable == 'DL':
        D = cosmo.angular_diameter_distance(z_L).value if DL is None else DL
    else:  # 'DS'
        D = cosmo.angular_diameter_distance(z_S).value if DS is None else DS
    D = np.asarray(D, dtype=float)
    return D / np.mean(D)


def gamma_covariate(variable, z_S, z_L, theta_E, velDisp=None):
    """Normalised lens property x used in a varied run: gamma = gamma_0 + gamma_1 * x.

    Two normalisations are used (kept as in the varied-gamma script):

    * 'z_S', 'z_L', 'theta_E', 'velDisp':  x = (q - mean(q)) / (max(q) - min(q)),
      with q = 1 + z for the redshifts. x is zero for the average lens, so gamma_0 is
      gamma at the sample mean and gamma_1 is the change in gamma across the full range.
    * 'DLS', 'DL', 'DS':  x = D / mean(D). x is 1 (not 0) for the average lens, so
      gamma at the sample mean is gamma_0 + gamma_1, and gamma_0 is the value
      extrapolated to zero distance.

    The mean, max and min are taken over the arrays passed in, so always pass the whole
    sample: a subset or a single lens gives a different (or undefined) normalisation.

    Args:
        variable (str): one of GAMMA_VARIABLES
        z_S (array): source redshifts
        z_L (array): lens redshifts
        theta_E (array): Einstein radii (in radians)
        velDisp (array, optional): velocity dispersions; only needed for variable='velDisp'

    Returns:
        array: x for each lens.
    """
    return _covariate(variable, z_S, z_L, theta_E, velDisp)


def vel(z_S, z_L, theta_E, seeing_atm, theta_ap, alpha, beta, delta, gamma,
        gamma_1=None, variable=None, velDisp=None):
    """Eq. (23) from arXiv:0907.4992v2

    Args:
        z_S (float): source redshift
        z_L (float): lens redshift
        theta_E (float): Einstein radius (in radians)
        seeing_atm (float): Atmospheric seeing (in radians)
        theta_ap (float): Aperture size (in radians)
        alpha (float): power-law matter density profile index
        beta (float): anisotropy parameter
        delta (float): luminosity density profile index
        gamma (float): slip parameter. In a varied run this is gamma_0.
        gamma_1 (float, optional): slope of gamma with `variable`. None (default) keeps
            gamma constant.
        variable (str, optional): lens property gamma varies with, one of GAMMA_VARIABLES.
            Required when gamma_1 is given.
        velDisp (array, optional): measured velocity dispersions; only needed when
            variable='velDisp'.

    Returns:
        float: analytic model for velocity dispersion (km/s).
    """

    # Angular diameter distances (plain floats, Mpc):
    DS = cosmo.angular_diameter_distance(z_S).value
    DLS = cosmo.angular_diameter_distance_z1z2(z_L, z_S).value

    if gamma_1 is not None:
        if variable is None:
            raise ValueError("gamma_1 was given without `variable`: choose one of "
                             f"{GAMMA_VARIABLES}")
        gamma = gamma + gamma_1 * _covariate(variable, z_S, z_L, theta_E, velDisp,
                                             DS=DS, DLS=DLS)

    # Unphysical parameter values can divide by zero; those points come out as
    # inf/nan and are rejected in log_likelihood, so the warnings are silenced here.
    with np.errstate(all='ignore'):
        # \chi
        chi = theta_ap/seeing_atm

        # \chi^tilde
        tilde_sigma = seeing_atm * \
            np.sqrt(1 + (chi ** 2) / 4 + (chi ** 4) / 40)  # Eq. (20)

        ksi = delta + alpha - 2

        term_1 = (2 / (1 + gamma)) * (clight ** 2 / 4) * (DS / DLS) * theta_E
        term_2 = (2 / np.sqrt(np.pi)) * ((2 * ((tilde_sigma / theta_E) ** 2))
                                         ** (1 - alpha / 2) / (ksi - 2 * beta))
        term_3 = (ratio_gamma(ksi) - beta * ratio_gamma(ksi + 2)) / \
            (ratio_gamma(alpha) * ratio_gamma(delta))
        term_4 = math.gamma((3 - ksi) / 2) / \
            math.gamma((3 - delta) / 2)

        sigma_star = term_1 * term_2 * term_3 * term_4

        return np.sqrt(np.abs(sigma_star))


# Goodness of fit of a statistical model


def log_likelihood(theta, z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap,
                   variable=None):
    """log(Eq. (25)) from arXiv:0907.4992v2

    Args:
        theta (list): [alpha, beta, delta, gamma] for a constant-gamma run, or
            [alpha, beta, delta, gamma_0, gamma_1] for a varied run
        z_S (float): source redshift
        z_L (float): lens redshift
        velDisp (float): velocity dispersion
        velDispErr (float): velocity dispersion error (std. dev.)
        theta_E (float): Einstein radius (in radians)
        seeing_atm (float): Atmospheric seeing (in radians)
        theta_ap (float): Aperture size (in radians)
        variable (str, optional): lens property gamma varies with (varied run only)

    Returns:
        float: Log-likehood function given model for velocity dispersion and the one measured.
            -inf if the model is not finite for these parameters.
    """
    alpha, beta, delta, gamma_0 = theta[0], theta[1], theta[2], theta[3]
    gamma_1 = theta[4] if len(theta) == 5 else None

    model = vel(z_S, z_L, theta_E, seeing_atm, theta_ap, alpha, beta, delta, gamma_0,
                gamma_1=gamma_1, variable=variable, velDisp=velDisp)
    LL = - 0.5*np.sum((velDisp - model) ** 2 / (velDispErr ** 2) + np.log(2 * np.pi * velDispErr ** 2))
    if not np.isfinite(LL):
        return - np.inf
    return LL


def log_prior(theta, alpha_0, eps_alpha_0, beta_0, eps_beta_0, delta_0, eps_delta_0,
              n_sigma=None):
    """Gaussian priors on alpha, beta and delta (gamma, gamma_0 and gamma_1 have none)

    Args:
        theta (list): [alpha, beta, delta, gamma] or [alpha, beta, delta, gamma_0, gamma_1]
        alpha_0 (array): expected value for alpha, one entry per lens
        eps_alpha_0 (array): prior width (std. dev.) of alpha, one entry per lens
        beta_0 (array): expected value for beta
        eps_beta_0 (array): prior width (std. dev.) of beta
        delta_0 (array): expected value for delta
        eps_delta_0 (array): prior width (std. dev.) of delta
        n_sigma (float, optional): half-width of the hard box around each prior mean, in
            units of the prior width. Default: 15 for a constant-gamma run and 5 for a
            varied run (the values used in the two original scripts).

    Returns:
        float: Sum of log of priors for alpha, beta, and delta.
    """
    alpha, beta, delta = theta[0], theta[1], theta[2]
    if n_sigma is None:
        n_sigma = 5 if len(theta) == 5 else 15

    if (alpha_0[0] - n_sigma * eps_alpha_0[0] < alpha < alpha_0[0] + n_sigma * eps_alpha_0[0]) and \
            (beta_0[0] - n_sigma * eps_beta_0[0] < beta < beta_0[0] + n_sigma * eps_beta_0[0]) and \
        (delta_0[0] - n_sigma * eps_delta_0[0] < delta < delta_0[0] + n_sigma * eps_delta_0[0]):
        log_prior_alpha = - 0.5 * \
            np.sum((alpha - alpha_0)**2 / eps_alpha_0 **
                   2 + np.log(2 * np.pi * eps_alpha_0**2))
        log_prior_beta = - 0.5 * \
            np.sum((beta - beta_0) ** 2 / eps_beta_0 **
                   2 + np.log(2 * np.pi * eps_beta_0**2))
        log_prior_delta = - 0.5 * \
            np.sum((delta - delta_0) ** 2 / eps_delta_0 **
                   2 + np.log(2 * np.pi * eps_delta_0**2))
        return log_prior_alpha + log_prior_beta + log_prior_delta
    else:
        return - np.inf


def log_probability(theta, z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap,
                    alpha_0, eps_alpha_0, beta_0, eps_beta_0, delta_0, eps_delta_0,
                    variable=None, n_sigma=None):
    """Log of probability of interest

    Args:
        theta (list): [alpha, beta, delta, gamma] or [alpha, beta, delta, gamma_0, gamma_1]
        z_S (float): source redshift
        z_L (float): lens redshift
        velDisp (float): velocity dispersion
        velDispErr (float): velocity dispersion error (std. dev.)
        theta_E (float): Einstein radius (in radians)
        seeing_atm (float): Atmospheric seeing (in radians)
        theta_ap (float): Aperture size (in radians)
        alpha_0, eps_alpha_0, beta_0, eps_beta_0, delta_0, eps_delta_0 (arrays):
            prior means and widths, see log_prior
        variable (str, optional): lens property gamma varies with (varied run only)
        n_sigma (float, optional): see log_prior

    Returns:
        float: Eq. (27) from arXiv:0907.4992v2
    """
    lp = log_prior(theta, alpha_0, eps_alpha_0, beta_0,
                   eps_beta_0, delta_0, eps_delta_0, n_sigma=n_sigma)
    if not np.isfinite(lp):
        return - np.inf
    else:
        ll = log_likelihood(theta, z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap,
                            variable=variable)
        return lp + ll


# Helpers shared by the minimization and sampling functions


def _prior_arrays(n, alpha_0_value, eps_alpha_0_value, beta_0_value, eps_beta_0_value,
                  delta_0_value, eps_delta_0_value):
    """Prior means and widths repeated once per lens, in the order log_prior expects."""
    return tuple(np.repeat(v, n) for v in (alpha_0_value, eps_alpha_0_value,
                                           beta_0_value, eps_beta_0_value,
                                           delta_0_value, eps_delta_0_value))


def _resolve_mode(gamma_ini, gamma_0_ini, gamma_1_ini, variable):
    """Returns (gamma start value, variable to use, True if this is a varied run)."""
    if gamma_0_ini is not None:
        gamma_ini = gamma_0_ini
    varied = gamma_1_ini is not None
    if varied:
        if variable is None:
            raise ValueError("gamma_1_ini was given without `variable`: choose one of "
                             f"{GAMMA_VARIABLES}")
        if variable not in GAMMA_VARIABLES:
            raise ValueError(f"variable must be one of {GAMMA_VARIABLES}, got {variable!r}")
    elif variable is not None:
        warnings.warn(f"variable={variable!r} is ignored because gamma_1_ini is None: "
                      "running with constant gamma. Pass gamma_1_ini (e.g. 0.0) for a "
                      "varied run.", stacklevel=3)
        variable = None
    return gamma_ini, variable, varied


# Minimizations and sampling methods


def minimization_loglikelihood(z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap,
                               seed=42, alpha_ini=2.0, beta_ini=0.18, delta_ini=2.4, gamma_ini=1.0,
                               gamma_1_ini=None, variable=None, gamma_0_ini=None):
    """Maximization of Likehood function

    Args:
        z_S (float): source redshift
        z_L (float): lens redshift
        velDisp (float): velocity dispersion
        velDispErr (float): velocity dispersion error (std. dev.)
        theta_E (float): Einstein radius (in radians)
        seeing_atm (float): Atmospheric seeing (in radians)
        theta_ap (float): Aperture size (in radians)
        seed (float): random seed for reproducibility purposes. Default: 42.
        alpha_ini (float, optional): Initial guess for alpha. Default: 2.0.
        beta_ini (float, optional): Initial guess for beta. Default: 0.18.
        delta_ini (float, optional): Initial guess for delta. Default: 2.4.
        gamma_ini (float, optional): Initial guess for gamma (gamma_0 in a varied run).
            Default: 1.0.
        gamma_1_ini (float, optional): Initial guess for gamma_1. Giving it (0.0 counts)
            switches on the varied run. Default: None (constant gamma).
        variable (str, optional): lens property gamma varies with, one of GAMMA_VARIABLES.
        gamma_0_ini (float, optional): another name for gamma_ini.

    Returns:
        tuple: alpha, beta, delta, gamma (and gamma_1 in a varied run) that maximise the
            likelihood.
    """
    gamma_ini, variable, varied = _resolve_mode(gamma_ini, gamma_0_ini, gamma_1_ini, variable)

    np.random.seed(seed)
    nll = lambda *args: - log_likelihood(*args)

    initial = [alpha_ini, beta_ini, delta_ini, gamma_ini] + ([gamma_1_ini] if varied else [])
    initial = np.array(initial) + 1e-5 * np.random.randn(len(initial))

    with np.errstate(all='ignore'):  # the optimiser may step through non-finite points
        soln = minimize(nll, initial, args=(z_S, z_L, velDisp, velDispErr, theta_E,
                        seeing_atm, theta_ap, variable))  # , method='Nelder-Mead', tol=1e-10)

    return tuple(soln.x)


def minimization_logprobability(z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap,
                                seed=42, alpha_ini=2.0, beta_ini=0.18, delta_ini=2.4, gamma_ini=1.0,
                                alpha_0_value=2.0, eps_alpha_0_value=0.08,
                                beta_0_value=0.18, eps_beta_0_value=0.13,
                                delta_0_value=2.4, eps_delta_0_value=0.11,
                                variable=None, gamma_1_ini=None, gamma_0_ini=None, n_sigma=None):
    """Maximization of Log Probability function

    Args:
        z_S (float): source redshift
        z_L (float): lens redshift
        velDisp (float): velocity dispersion
        velDispErr (float): velocity dispersion error (std. dev.)
        theta_E (float): Einstein radius (in radians)
        seeing_atm (float): Atmospheric seeing (in radians)
        theta_ap (float): Aperture size (in radians)
        seed (float): random seed for reproducibility purposes.
        alpha_ini (float, optional): Initial guess for alpha. Defaults to 2.0.
        beta_ini (float, optional): Initial guess for beta. Defaults to 0.18.
        delta_ini (float, optional): Initial guess for delta. Defaults to 2.4.
        gamma_ini (float, optional): Initial guess for gamma (gamma_0 in a varied run).
            Defaults to 1.0.
        alpha_0_value (float, optional): expected value for alpha. Defaults to 2.0.
        eps_alpha_0_value (float, optional): prior width (std. dev.) for alpha. Defaults to 0.08.
        beta_0_value (float, optional): expected value for beta. Defaults to 0.18.
        eps_beta_0_value (float, optional): prior width (std. dev.) for beta. Defaults to 0.13.
        delta_0_value (float, optional): expected value for delta. Defaults to 2.4.
        eps_delta_0_value (float, optional): prior width (std. dev.) for delta. Defaults to 0.11.
        variable (str, optional): lens property gamma varies with, one of GAMMA_VARIABLES.
        gamma_1_ini (float, optional): Initial guess for gamma_1. Giving it (0.0 counts)
            switches on the varied run. Defaults to None (constant gamma).
        gamma_0_ini (float, optional): another name for gamma_ini.
        n_sigma (float, optional): half-width of the hard prior box, see log_prior.

    Returns:
        list: alpha, beta, delta, gamma (and gamma_1 in a varied run) that maximise the
            log-probability.
    """
    gamma_ini, variable, varied = _resolve_mode(gamma_ini, gamma_0_ini, gamma_1_ini, variable)

    priors = _prior_arrays(len(z_S), alpha_0_value, eps_alpha_0_value, beta_0_value,
                           eps_beta_0_value, delta_0_value, eps_delta_0_value)

    np.random.seed(seed)
    nll_2 = lambda *args: - log_probability(*args)

    initial = [alpha_ini, beta_ini, delta_ini, gamma_ini] + ([gamma_1_ini] if varied else [])
    initial = np.array(initial) + 1e-5 * np.random.randn(len(initial))

    with np.errstate(all='ignore'):  # the optimiser may step outside the prior box (-inf)
        soln_2 = minimize(nll_2, initial, args=(z_S, z_L, velDisp, velDispErr, theta_E,
                          seeing_atm, theta_ap, *priors, variable, n_sigma))  # , method='Nelder-Mead', tol=1e-10)

    return [float(x) for x in soln_2.x]


def logprobability_sampling(z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap,
                            seed=42, alpha_ini=2.0, beta_ini=0.18, delta_ini=2.4, gamma_ini=1.0,
                            alpha_0_value=2.0, eps_alpha_0_value=0.08,
                            beta_0_value=0.18, eps_beta_0_value=0.13,
                            delta_0_value=2.4, eps_delta_0_value=0.11,
                            n_dim=None, n_walkers=64, n_burn=500, n_steps=10000, progress=True,
                            processes=1, variable=None, gamma_1_ini=None, gamma_0_ini=None,
                            n_sigma=None, p0_scatter=None):
    """Sampling logprobability function with emcee

    Args:
        z_S (float): source redshift
        z_L (float): lens redshift
        velDisp (float): velocity dispersion
        velDispErr (float): velocity dispersion error (std. dev.)
        theta_E (float): Einstein radius (in radians)
        seeing_atm (float): Atmospheric seeing (in radians)
        theta_ap (float): Aperture size (in radians)
        seed (float): random seed. Fixes the walkers' starting points and the chain itself,
            so the same seed gives the same chain.
        alpha_ini (float, optional): Initial guess for alpha. Defaults to 2.0.
        beta_ini (float, optional): Initial guess for beta. Defaults to 0.18.
        delta_ini (float, optional): Initial guess for delta. Defaults to 2.4.
        gamma_ini (float, optional): Initial guess for gamma (gamma_0 in a varied run).
            Defaults to 1.0.
        alpha_0_value (float, optional): expected value for alpha. Defaults to 2.0.
        eps_alpha_0_value (float, optional): prior width (std. dev.) for alpha. Defaults to 0.08.
        beta_0_value (float, optional): expected value for beta. Defaults to 0.18.
        eps_beta_0_value (float, optional): prior width (std. dev.) for beta. Defaults to 0.13.
        delta_0_value (float, optional): expected value for delta. Defaults to 2.4.
        eps_delta_0_value (float, optional): prior width (std. dev.) for delta. Defaults to 0.11.
        n_dim (int, optional): number of parameters in the model. Defaults to None, which
            means 4 for a constant-gamma run and 5 for a varied run. Passing the matching
            number explicitly also works; a mismatch raises an error.
        n_walkers (int, optional): number of MCMC walkers. Defaults to 64.
        n_burn (int, optional): "burn-in" period to let chains stabilize. Defaults to 500.
        n_steps (int, optional): number of MCMC steps to take after burn-in. Defaults to 10000.
        progress (bool, optional): Show progress bar. Defaults to True.
        processes (int, optional): Number of processes in parallel. Defaults to 1 (serial).
        variable (str, optional): lens property gamma varies with, one of GAMMA_VARIABLES
            ('z_S', 'z_L', 'theta_E', 'velDisp', 'DLS', 'DL', 'DS'). See gamma_covariate.
        gamma_1_ini (float, optional): Initial guess for gamma_1. Giving it (0.0 counts)
            switches on the varied run. Defaults to None (constant gamma).
        gamma_0_ini (float, optional): another name for gamma_ini.
        n_sigma (float, optional): half-width of the hard prior box, see log_prior.
            Defaults to 15 (constant gamma) or 5 (varied).
        p0_scatter (float, optional): size of the ball the walkers start in, around the
            maximum-likelihood point. Defaults to 1e-3 (constant gamma) or 1e-2 (varied).

    Returns:
        emcee.EnsembleSampler: the sampler after the production run. The chain columns
            are [alpha, beta, delta, gamma] or [alpha, beta, delta, gamma_0, gamma_1].
    """
    gamma_ini, variable, varied = _resolve_mode(gamma_ini, gamma_0_ini, gamma_1_ini, variable)

    n_par = 5 if varied else 4
    if n_dim is None:
        n_dim = n_par
    elif n_dim != n_par:
        raise ValueError(f"n_dim={n_dim} does not match this run, which has {n_par} parameters "
                         f"({'varied' if varied else 'constant'} gamma). Leave n_dim out to "
                         "have it set automatically.")
    if p0_scatter is None:
        p0_scatter = 1e-2 if varied else 1e-3

    priors = _prior_arrays(len(z_S), alpha_0_value, eps_alpha_0_value, beta_0_value,
                           eps_beta_0_value, delta_0_value, eps_delta_0_value)
    args = (z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap, *priors,
            variable, n_sigma)

    pool = Pool(processes=processes) if processes and processes > 1 else None
    try:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, args=args, pool=pool)

        np.random.seed(seed)
        solu = minimization_loglikelihood(z_S, z_L, velDisp, velDispErr, theta_E, seeing_atm, theta_ap,
                                          seed, alpha_ini, beta_ini, delta_ini, gamma_ini,
                                          gamma_1_ini, variable)
        p0 = solu + p0_scatter * np.random.randn(n_walkers, n_dim)

        # emcee otherwise seeds its own generator from the operating system, which made
        # the chains differ from run to run even with the same `seed`.
        sampler.random_state = np.random.get_state()

        if not np.isfinite(log_probability(solu, *args)):
            warnings.warn("The maximum-likelihood starting point "
                          f"{np.round(solu, 3)} has zero prior probability (outside the "
                          "n_sigma box), so the walkers may not move. Check the acceptance "
                          "fraction, or widen n_sigma.", stacklevel=2)

        # Run n_burn steps as a burn-in:
        print('Running burn-in ...')
        pos, prob, state = sampler.run_mcmc(p0, n_burn, progress=progress)

        # Reset the chain to remove the burn-in samples:
        sampler.reset()

        # Starting from the final position in the burn-in chain, sample for n_steps steps:
        print('Sampling ...')
        sampler.run_mcmc(pos, n_steps, rstate0=state, progress=progress)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return sampler
    
    
def run_group_optimization(divided_data, priors, starts, seed=11):
    """
    Runs MAP and ML optimization for each group.
    
    Returns:
        dict: A dictionary of optimization results mapping group_name to (X_map, X_ml).
    """
    results = {}
    for group_name, sub_df in divided_data.items():
        print(f"\n--- Optimizing Group: {group_name} (Lenses: {len(sub_df)}) ---")
        group_args = (
            sub_df["z_S"].to_numpy(dtype=float),
            sub_df["z_L"].to_numpy(dtype=float),
            sub_df["velDisp"].to_numpy(dtype=float),
            sub_df["velDispErr"].to_numpy(dtype=float),
            sub_df["theta_E_rad"].to_numpy(dtype=float),
            sub_df["Seeing_rad"].to_numpy(dtype=float),
            sub_df["theta_ap_rad"].to_numpy(dtype=float)
        )
        try:
            X_map = rings2cosmo.minimization_logprobability(*group_args, seed=seed, **starts, **priors)
            X_ml = rings2cosmo.minimization_loglikelihood(*group_args, seed=seed, **starts)
            print(f"  MAP parameters: alpha={X_map[0]:.3f}, beta={X_map[1]:.3f}, delta={X_map[2]:.3f}, gamma={X_map[3]:.3f}")
            print(f"  ML parameters:  alpha={X_ml[0]:.3f}, beta={X_ml[1]:.3f}, delta={X_ml[2]:.3f}, gamma={X_ml[3]:.3f}")
            results[group_name] = (X_map, X_ml)
        except Exception as e:
            print(f"  Error optimizing {group_name}: {e}")
    return results
    
def run_group_sampling(divided_data, priors, starts, n_dim=4, n_walkers=100, n_burn=500, n_steps=5000, seed=11):
    """
    Runs MCMC sampler for each group.
    
    Returns:
        dict: A dictionary mapping group_name to its completed emcee sampler object.
    """
    samplers = {}
    for group_name, sub_df in divided_data.items():
        print(f"\n--- Sampling Group: {group_name} (Lenses: {len(sub_df)}) ---")
        group_args = (
            sub_df["z_S"].to_numpy(dtype=float),
            sub_df["z_L"].to_numpy(dtype=float),
            sub_df["velDisp"].to_numpy(dtype=float),
            sub_df["velDispErr"].to_numpy(dtype=float),
            sub_df["theta_E_rad"].to_numpy(dtype=float),
            sub_df["Seeing_rad"].to_numpy(dtype=float),
            sub_df["theta_ap_rad"].to_numpy(dtype=float)
        )
        try:
            sampler = rings2cosmo.logprobability_sampling(
                *group_args, seed=seed, **starts, **priors,
                n_dim=n_dim, n_walkers=n_walkers, n_burn=n_burn, n_steps=n_steps,
                progress=True, processes=None
            )
            samplers[group_name] = sampler
        except Exception as e:
            print(f"  Error sampling {group_name}: {e}")
    return samplers
    
def run_group_diagnostics(samplers, labels=None):
    """
    Plots trace lines and prints chain diagnostics for each active group sampler.
    """
    if labels is None:
        labels = [r"$\alpha$", r"$\beta$", r"$\delta$", r"$\gamma$"]
    n_dim = len(labels)
    
    for group_name, sampler in samplers.items():
        print(f"\n=== Diagnostics for {group_name} ===")
        acc = sampler.acceptance_fraction
        print(f"  Mean acceptance fraction: {acc.mean():.3f} (min {acc.min():.3f}, max {acc.max():.3f})")
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            print(f"  Autocorrelation times: {np.round(tau, 1)}")
        except Exception as e:
            print(f"  Could not calculate autocorrelation: {e}")
            
        samples = sampler.get_chain()
        fig, axes = plt.subplots(n_dim, figsize=(10, 5), sharex=True)
        fig.suptitle(f"Trace plots: {group_name}")
        for i in range(n_dim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.2, lw=0.5)
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("Step number")
        plt.show()
        
def plot_group_gamma_results(samplers, labels=None):
    """
    Plots a summary comparison of gamma (PPN slip parameter) posteriors across all groups.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    group_names = []
    medians = []
    err_downs = []
    err_ups = []
    
    for i, (group_name, sampler) in enumerate(samplers.items()):
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            discard = int(2 * np.max(tau))
            thin = max(1, int(0.5 * np.min(tau)))
        except:
            discard = 100
            thin = 15
            
        flat = sampler.get_chain(discard=discard, thin=thin, flat=True)
        gamma_samples = flat[:, 3] # gamma is the 4th parameter index
        
        lo, med, hi = np.percentile(gamma_samples, [16, 50, 84])
        group_names.append(group_name)
        medians.append(med)
        err_downs.append(med - lo)
        err_ups.append(hi - med)
        
    y_pos = np.arange(len(group_names))
    ax.errorbar(medians, y_pos, xerr=[err_downs, err_ups], fmt='o', color='blue', capsize=5, label='Posterior 68% CI')
    ax.axvline(1.0, color='red', linestyle='--', label='GR Expectation (? = 1)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_names)
    ax.invert_yaxis() # lists top-down
    ax.set_xlabel(r'Slip Parameter $\gamma$')
    ax.set_title(r'Comparison of PPN parameter $\gamma$ by Group')
    ax.legend()
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.show()
