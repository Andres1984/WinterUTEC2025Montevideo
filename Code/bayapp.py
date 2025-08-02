import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm, invgamma
from scipy.special import comb

st.set_page_config(page_title="Bayesian Visualizer", layout="wide")
st.markdown("# 📊 Bayesian Distribution Visualizer")

col_sidebar, col_main = st.columns([1, 3], gap="large")

with col_sidebar:
    model = st.selectbox("Select a Bayesian Model", [
        "Beta-Binomial", "Gamma-Poisson", "Normal-Normal", "Normal with Unknown Variance"
    ])

    st.markdown("### 🔧 Adjust Model Parameters")

    theta = np.linspace(0.001, 0.999, 500)
    mu_grid = np.linspace(-10, 10, 500)
    sigma2_grid = np.linspace(0.1, 20, 500)

    figs = []

    if model == "Beta-Binomial":
        alpha = st.slider("α (prior)", 0.1, 100.0, 2.0)
        beta_param = st.slider("β (prior)", 0.1, 100.0, 2.0)
        k = st.slider("Successes (k)", 0, 50, 5)
        n = st.slider("Trials (n)", k, 50, 10)

        prior = beta.pdf(theta, alpha, beta_param)
        likelihood = comb(n, k) * (theta ** k) * ((1 - theta) ** (n - k))
        likelihood = likelihood / np.max(likelihood) * np.max(prior)
        posterior = beta.pdf(theta, alpha + k, beta_param + n - k)

        for dist, label, color in zip([prior, likelihood, posterior],
                                      ["Prior: Beta", "Likelihood: Binomial (scaled)", "Posterior: Beta"],
                                      ["blue", "orange", "green"]):
            fig, ax = plt.subplots()
            ax.plot(theta, dist, lw=2, color=color)
            ax.set_title(label)
            ax.set_xlabel("θ")
            ax.set_ylabel("Density")
            ax.grid(True)
            figs.append(fig)

    elif model == "Gamma-Poisson":
        alpha = st.slider("α (prior)", 0.1, 50.0, 2.0)
        beta_param = st.slider("β (prior)", 0.1, 20.0, 1.0)
        total_counts = st.slider("Total counts (∑x)", 0, 100, 12)
        n_obs = st.slider("Observations (n)", 1, 50, 5)

        theta = np.linspace(0.001, 20, 500)
        prior = gamma.pdf(theta, a=alpha, scale=1 / beta_param)
        likelihood = (theta ** total_counts) * np.exp(-n_obs * theta)
        likelihood /= np.max(likelihood) * np.max(prior)
        posterior = gamma.pdf(theta, a=alpha + total_counts, scale=1 / (beta_param + n_obs))

        for dist, label, color in zip([prior, likelihood, posterior],
                                      ["Prior: Gamma", "Likelihood: Poisson (scaled)", "Posterior: Gamma"],
                                      ["blue", "orange", "green"]):
            fig, ax = plt.subplots()
            ax.plot(theta, dist, lw=2, color=color)
            ax.set_title(label)
            ax.set_xlabel("θ")
            ax.set_ylabel("Density")
            ax.grid(True)
            figs.append(fig)

    elif model == "Normal-Normal":
        mu0 = st.slider("μ₀ (prior mean)", -10.0, 10.0, 0.0)
        tau = st.slider("τ (prior std)", 0.1, 10.0, 2.0)
        sigma = st.slider("σ (known std)", 0.1, 10.0, 1.0)
        n = st.slider("Number of observations", 1, 100, 10)
        xbar = st.slider("Sample mean x̄", -10.0, 10.0, 1.0)

        prior = norm.pdf(mu_grid, mu0, tau)
        likelihood = norm.pdf(mu_grid, loc=xbar, scale=np.sqrt(sigma**2 / n))
        likelihood = likelihood / np.max(likelihood) * np.max(prior)
        post_var = 1 / (1 / tau**2 + n / sigma**2)
        post_mean = post_var * (mu0 / tau**2 + n * xbar / sigma**2)
        posterior = norm.pdf(mu_grid, loc=post_mean, scale=np.sqrt(post_var))

        for dist, label, color in zip([prior, likelihood, posterior],
                                      ["Prior: Normal", "Likelihood: Normal (scaled)", "Posterior: Normal"],
                                      ["blue", "orange", "green"]):
            fig, ax = plt.subplots()
            ax.plot(mu_grid, dist, lw=2, color=color)
            ax.set_title(label)
            ax.set_xlabel("μ")
            ax.set_ylabel("Density")
            ax.grid(True)
            figs.append(fig)

    elif model == "Normal with Unknown Variance":
        mu0 = st.slider("μ₀ (prior mean)", -10.0, 10.0, 0.0)
        kappa0 = st.slider("κ₀ (prior precision)", 0.1, 10.0, 1.0)
        alpha = st.slider("α (IG prior)", 0.1, 10.0, 2.0)
        beta_param = st.slider("β (IG prior)", 0.1, 20.0, 2.0)
        xbar = st.slider("Sample mean x̄", -10.0, 10.0, 1.0)
        s2 = st.slider("Sample variance s²", 0.1, 10.0, 2.0)
        n = st.slider("Observations n", 1, 100, 10)

        alpha_post = alpha + n / 2
        beta_post = beta_param + 0.5 * (n - 1) * s2 + (kappa0 * n * (xbar - mu0)**2) / (2 * (kappa0 + n))
        kappa_post = kappa0 + n
        mu_post = (kappa0 * mu0 + n * xbar) / kappa_post

        sigma2_mean_post = beta_post / (alpha_post - 1) if alpha_post > 1 else beta_post / 1e-3
        prior_mu = norm.pdf(mu_grid, loc=mu0, scale=np.sqrt(beta_param / (alpha - 1) / kappa0))
        likelihood = norm.pdf(mu_grid, loc=xbar, scale=np.sqrt(s2 / n))
        likelihood = likelihood / np.max(likelihood) * np.max(prior_mu)
        posterior_mu = norm.pdf(mu_grid, loc=mu_post, scale=np.sqrt(sigma2_mean_post / kappa_post))

        prior_sigma2 = invgamma.pdf(sigma2_grid, a=alpha, scale=beta_param)
        posterior_sigma2 = invgamma.pdf(sigma2_grid, a=alpha_post, scale=beta_post)

        # Combine prior and posterior of variance in the same plot
        fig, ax = plt.subplots()
        ax.plot(sigma2_grid, prior_sigma2, lw=2, label="Prior σ²", color="blue")
        ax.plot(sigma2_grid, posterior_sigma2, lw=2, label="Posterior σ²", color="green")
        ax.set_title("Prior and Posterior: σ²")
        ax.set_xlabel("σ²")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()
        figs.append(fig)

        for dist, label, color in zip([prior_mu, likelihood, posterior_mu],
                                      ["Prior: μ", "Likelihood: Normal (scaled)", "Posterior: μ"],
                                      ["blue", "orange", "green"]):
            fig, ax = plt.subplots()
            ax.plot(mu_grid, dist, lw=2, color=color)
            ax.set_title(label)
            ax.set_xlabel("μ")
            ax.set_ylabel("Density")
            ax.grid(True)
            figs.append(fig)

with col_main:
    st.markdown("### 📈 Visualizations")
    rows = 2
    cols_per_row = 2
    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(figs):
                with cols[j]:
                    st.pyplot(figs[idx])
            else:
                with cols[j]:
                    st.empty()

