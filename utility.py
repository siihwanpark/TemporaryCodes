import argparse
import numpy as np

from scipy.stats import t, norm, chi2, f
from statsmodels.stats.libqsturng import psturng
from math import sqrt, floor

def parse_option():
    parser = argparse.ArgumentParser('Arguments for Usage')

    parser.add_argument('--type', type=str,
                        help='utility type')
    
    opt = parser.parse_args()

    return opt

"""
Helper functions
"""

def t_alpha_(alpha, nu):
    t_dist = t(nu)
    
    return t_dist.ppf(alpha)

def z_alpha_(alpha):
    std_norm = norm(0,1)

    return std_norm.ppf(alpha)

"""
Chapter 8. Inferences on a Population Mean
"""

def t_intervals(x_var, s, n, alpha, L=None):

    t_alpha = t_alpha_(1-(alpha/2), n-1)
    print("Two-Sided t-interval with x_var {}, s {}, n {}, α {}:".
            format(x_var, s, n, alpha))
    print("({:.3f}, {:.3f})".format(x_var - t_alpha*s/sqrt(n), x_var + t_alpha*s/sqrt(n)))
    print("========================================================================")

    t_alpha = t_alpha_(1-alpha, n-1)
    print("One-Sided t-intervals with x_var {}, s {}, n {}, α {}:".
            format(x_var, s, n, alpha))
    print("Upper : (-∞, {:.3f})".format(x_var + t_alpha*s/sqrt(n)))
    print("Lower : ({:.3f}, ∞)".format(x_var - t_alpha*s/sqrt(n)))
    print("========================================================================")

    if L is not None:
        print("Length of C.I. <= {} requires n >= {:.1f} ".format(L, (2*t_alpha*s/L)**2))
        print("========================================================================")

# population variance is known                                                                                                                                                                                                                                                                                                                                    
def z_intervals(x_var, sigma, n, alpha, L=None):
    z_alpha = z_alpha_(1-(alpha/2))

    print("Two-Sided z-interval with x_var {}, σ {}, n {}, α {}:".
            format(x_var, sigma, n, alpha))
    print("({:.3f}, {:.3f})".format(x_var - z_alpha*sigma/sqrt(n), x_var + z_alpha*sigma/sqrt(n)))
    print("========================================================================")

    z_alpha = z_alpha_(1-alpha)

    print("One-Sided z-intervals with x_var {}, σ {}, n {}, α {}:".
            format(x_var, sigma, n, alpha))
    print("Upper : (-∞, {:.3f})".format(x_var + z_alpha*sigma/sqrt(n)))
    print("Lower : ({:.3f}, ∞)".format(x_var - z_alpha*sigma/sqrt(n)))
    print("========================================================================")

    if L is not None:
        print("Length of C.I. <= {} requires n >= {:.1f} ".format(L, (2*z_alpha*sigma/L)**2))
        print("========================================================================")

def t_tests_on_mean(mu_0, x_var, s, n, alpha, power=None):
    print("Two-Sided t-Test - H_0 : μ = {} vs H_A : μ ≠ {}".format(mu_0, mu_0))
    print("with x_var {}, s {}, n {}, α {} :\n".format(x_var, s, n, alpha))

    t_statistic = (x_var - mu_0)/(s/sqrt(n))
    p_value = t.sf(np.abs(t_statistic), n-1)*2
    print("t-statistic : {:.4f}, p-value : 2P(T>=|t|) = {:.3f}".format(t_statistic, p_value))
    print("The null hypothesis is {}".format("Accepted" if p_value > alpha else "Rejected"))
    print("========================================================================")

    print("One-Sided t-Test - H_0 : μ <= {} vs H_A : μ > {}".format(mu_0, mu_0))
    print("with x_var {}, s {}, n {}, α {} :\n".format(x_var, s, n, alpha))

    p_value = t.sf(t_statistic, n-1)
    print("t-statistic : {:.4f}, p-value : P(T>=t) = {:.3f}".format(t_statistic, p_value))
    print("The null hypothesis is {}".format("Accepted" if p_value > alpha else "Rejected"))
    print("========================================================================")

    print("One-Sided t-Test - H_0 : μ >= {} vs H_A : μ < {}".format(mu_0, mu_0))
    print("with x_var {}, s {}, n {}, α {} :\n".format(x_var, s, n, alpha))

    p_value = 1 - t.sf(t_statistic, n-1)
    print("t-statistic : {:.4f}, p-value : P(T<=t) = {:.3f}".format(t_statistic, p_value))
    print("The null hypothesis is {}".format("Accepted" if p_value > alpha else "Rejected"))
    print("========================================================================")

    if power is not None:
        raise NotImplementedError
        print("Power >= {} requires n >= {}".format(power, 1))
        print("========================================================================")

# population variance is known
def z_tests_on_mean(mu_0, x_var, sigma, n, alpha, power=None):
    print("Two-Sided z-Test - H_0 : μ = {} vs H_A : μ ≠ {}".format(mu_0, mu_0))
    print("with x_var {}, σ {}, n {}, α {} :\n".format(x_var, sigma, n, alpha))

    z_statistic = (x_var - mu_0)/(sigma/sqrt(n))
    p_value = norm.sf(np.abs(z_statistic))*2
    print("z-statistic : {:.4f}, p-value : 2P(Z>=|z|) = {:.3f}".format(z_statistic, p_value))
    print("The null hypothesis is {}".format("Accepted" if p_value > alpha else "Rejected"))
    print("========================================================================")

    print("One-Sided z-Test - H_0 : μ <= {} vs H_A : μ > {}".format(mu_0, mu_0))
    print("with x_var {}, σ {}, n {}, α {} :\n".format(x_var, sigma, n, alpha))

    p_value = norm.sf(z_statistic)
    print("z-statistic : {:.4f}, p-value : P(Z>=z) = {:.3f}".format(z_statistic, p_value))
    print("The null hypothesis is {}".format("Accepted" if p_value > alpha else "Rejected"))
    print("========================================================================")

    print("One-Sided z-Test - H_0 : μ >= {} vs H_A : μ < {}".format(mu_0, mu_0))
    print("with x_var {}, σ {}, n {}, α {} :\n".format(x_var, sigma, n, alpha))

    p_value = 1- norm.sf(z_statistic)
    print("z-statistic : {:.4f}, p-value : P(Z<=z) = {:.3f}".format(z_statistic, p_value))
    print("The null hypothesis is {}".format("Accepted" if p_value > alpha else "Rejected"))
    print("========================================================================")

    if power is not None:
        raise NotImplementedError
        print("Power >= {} requires n >= {}".format(power, 1))
        print("========================================================================")

"""
Chapter 9. Comparing Two Population Means
"""

def two_indep_sample_problem(n, m, x_var, y_var, s_x, s_y, alpha, delta, type='general'):
    if type == 'general':
        std_err = sqrt((s_x**2)/n + (s_y**2)/m)
        nu_star = ((((s_x**2)/n) + ((s_y**2)/m))**2)/((s_x**4/((n**2)*(n-1))) + (s_y**4/((m**2)*(m-1))))
        nu = floor(nu_star)

        t_alpha = t_alpha_(1-(alpha/2), nu)
        print("Two Independent Samples : General Procedure")
        print("with x_var {}, y_var {}, n {}, m {}, s_x {}, s_y {}, alpha {}, delta {}".format(x_var, y_var, n, m, s_x, s_y, alpha, delta))
        print("========================================================================")

        print("s.e. : {:.3f}".format(std_err))
        print("degree of freedom : {:.4f} -> {}".format(nu_star, nu))
        print("========================================================================")

        print("Confidence Intervals of mu_A - mu_B with {:.2f} :".format(1-alpha))
        print("Two-Sided : ({:.3f}, {:.3f})".format(x_var-y_var-t_alpha*std_err, x_var-y_var+t_alpha*std_err))
        
        t_alpha = t_alpha_(1-alpha, nu)
        print("Upper : (-∞, {:.3f})".format(x_var-y_var+t_alpha*std_err))
        print("Lower : ({:.3f}, ∞)".format(x_var-y_var-t_alpha*std_err))
        print("========================================================================")

        print("Hypothesis Testing H_0 : mu_A - mu_B = {} vs H_A : mu_A - mu_B ≠ {}".format(delta, delta))
        print("========================================================================")
        
        t_statistics = (x_var-y_var-delta)/std_err
        print("t-statistic : {:.4f}".format(t_statistics))
        print("p_value = 2P(T>=|t|) : {:.3f}".format(t.sf(np.abs(t_statistics), nu)*2))
        print("========================================================================")

        print("Hypothesis Testing H_0 : mu_A - mu_B <= {} vs H_A : mu_A - mu_B > {}".format(delta, delta))
        print("========================================================================")
        
        t_statistics = (x_var-y_var-delta)/std_err
        print("t-statistic : {:.4f}".format(t_statistics))
        print("p_value = P(T>=t) : {:.3f}".format(t.sf(t_statistics, nu)))
        print("========================================================================")
        
        print("Hypothesis Testing H_0 : mu_A - mu_B >= {} vs H_A : mu_A - mu_B < {}".format(delta, delta))
        print("========================================================================")
        
        t_statistics = (x_var-y_var-delta)/std_err
        print("t-statistic : {:.4f}".format(t_statistics))
        print("p_value = P(T<=t) : {:.3f}".format(1-t.sf(t_statistics, nu)))
        print("========================================================================")

    elif type == 'pooled':
        s_p_square = ((n-1)*(s_x**2) + (m-1)*(s_y**2))/(n+m-2)
        std_err = sqrt(s_p_square*(1/n + 1/m))

        print("Two Independent Samples : Pooled Variance Procedure")
        print("with x_var {}, y_var {}, n {}, m {}, s_x {}, s_y {}, alpha {}, delta {}".format(x_var, y_var, n, m, s_x, s_y, alpha, delta))
        print("========================================================================")

        print("s_p^2 : {:.3f}, s_p : {:.3f}, s.e. : {:.3f}".format(s_p_square, sqrt(s_p_square), std_err))
        print("degree of freedom = n+m-2 : {}".format(n+m-2))
        print("========================================================================")

        t_alpha = t_alpha_(1-(alpha/2), n+m-2)
        print("Confidence Intervals of mu_A - mu_B with {:.2f} :".format(1-alpha))
        print("Two-Sided : ({:.3f}, {:.3f})".format(x_var-y_var-t_alpha*std_err, x_var-y_var+t_alpha*std_err))
        
        t_alpha = t_alpha_(1-alpha, n+m-2)
        print("Upper : (-∞, {:.3f})".format(x_var-y_var+t_alpha*std_err))
        print("Lower : ({:.3f}, ∞)".format(x_var-y_var-t_alpha*std_err))
        print("========================================================================")

        print("Hypothesis Testing H_0 : mu_A - mu_B = {} vs H_A : mu_A - mu_B ≠ {}".format(delta, delta))
        print("========================================================================")
        
        t_statistics = (x_var-y_var-delta)/std_err
        print("t-statistic : {:.4f}".format(t_statistics))
        print("p_value = 2P(T>=|t|) : {:.3f}".format(t.sf(np.abs(t_statistics), n+m-2)*2))
        print("========================================================================")

        print("Hypothesis Testing H_0 : mu_A - mu_B <= {} vs H_A : mu_A - mu_B > {}".format(delta, delta))
        print("========================================================================")
        
        t_statistics = (x_var-y_var-delta)/std_err
        print("t-statistic : {:.4f}".format(t_statistics))
        print("p_value = P(T>=t) : {:.3f}".format(t.sf(t_statistics, n+m-2)))
        print("========================================================================")
        
        print("Hypothesis Testing H_0 : mu_A - mu_B >= {} vs H_A : mu_A - mu_B < {}".format(delta, delta))
        print("========================================================================")
        
        t_statistics = (x_var-y_var-delta)/std_err
        print("t-statistic : {:.4f}".format(t_statistics))
        print("p_value = P(T<=t) : {:.3f}".format(1-t.sf(t_statistics, n+m-2)))
        print("========================================================================")

    elif type == 'z':
        print("Two Independent Samples : z-Procedure")
        print("with x_var {}, y_var {}, n {}, m {}, σ_A {}, σ_B {}, alpha {}, delta {}".format(x_var, y_var, n, m, s_x, s_y, alpha, delta))
        print("========================================================================")

        std_err = sqrt((s_x**2)/n + (s_y**2)/m)
        print("s.e. : {:.3f}".format(std_err))
        print("========================================================================")

        z_alpha = z_alpha_(1-(alpha/2))
        print("Confidence Intervals of mu_A - mu_B with {:.2f} :".format(1-alpha))
        print("Two-Sided : ({:.3f}, {:.3f})".format(x_var-y_var-z_alpha*std_err, x_var-y_var+z_alpha*std_err))
        
        z_alpha = z_alpha_(1-alpha)
        print("Upper : (-∞, {:.3f})".format(x_var-y_var+z_alpha*std_err))
        print("Lower : ({:.3f}, ∞)".format(x_var-y_var-z_alpha*std_err))
        print("========================================================================")

        print("Hypothesis Testing H_0 : mu_A - mu_B = {} vs H_A : mu_A - mu_B ≠ {}".format(delta, delta))
        print("========================================================================")
        
        z_statistics = (x_var-y_var-delta)/std_err
        print("z-statistic : {:.4f}".format(z_statistics))
        print("p_value = 2P(Z>=|z|) : {:.3f}".format(norm.sf(np.abs(z_statistics))*2))
        print("========================================================================")

        print("Hypothesis Testing H_0 : mu_A - mu_B <= {} vs H_A : mu_A - mu_B > {}".format(delta, delta))
        print("========================================================================")
        
        z_statistics = (x_var-y_var-delta)/std_err
        print("z-statistic : {:.4f}".format(z_statistics))
        print("p_value = P(Z>=z) : {:.3f}".format(norm.sf(z_statistics)))
        print("========================================================================")
        
        print("Hypothesis Testing H_0 : mu_A - mu_B >= {} vs H_A : mu_A - mu_B < {}".format(delta, delta))
        print("========================================================================")
        
        z_statistics = (x_var-y_var-delta)/std_err
        print("z-statistic : {:.4f}".format(z_statistics))
        print("p_value = P(Z<=z) : {:.3f}".format(1-norm.sf(z_statistics)))
        print("========================================================================")
    else:
        raise TypeError

"""
Chapter 10. Discrete Data Analysis
"""

def sample_proportion(n, x, p_0, alpha):
    p_hat = x/n
    z_alpha = z_alpha_(1-(alpha/2))
    print("p_hat : {:.3f}".format(p_hat))

    print("C.I. of p with alpha {}".format(alpha))
    print("Two-Sided : ({:.3f}, {:.3f})".format(p_hat-z_alpha*sqrt(p_hat*(1-p_hat)/n), p_hat+z_alpha*sqrt(p_hat*(1-p_hat)/n)))
    
    z_alpha = z_alpha_(1-alpha)
    print("Upper : ({:.3f}, 1)".format(p_hat-z_alpha*sqrt(p_hat*(1-p_hat)/n)))
    print("Lower : (0, {:.3f})".format(p_hat+z_alpha*sqrt(p_hat*(1-p_hat)/n)))
    print("========================================================================")

    print("Hypothesis Testing - H_0 : p = {} vs H_A : p ≠ {}".format(p_0, p_0))

    z_statistics = (p_hat - p_0)/sqrt(p_0*(1-p_0)/n)
    print("z-statistics : {:.4f}".format(z_statistics))
    print("p-value = 2P(Z <= -|z|) : {:.3f}".format(2*(1-norm.sf(-np.abs(z_statistics)))))
    print("========================================================================")
    
    print("Hypothesis Testing - H_0 : p >= {} vs H_A : p < {}".format(p_0, p_0))

    z_statistics = (x + 0.5 - n*p_0)/sqrt(n*p_0*(1-p_0))
    print("z-statistics : {:.4f}".format(z_statistics))
    print("p-value = P(Z <= z) : {:.3f}".format(1-norm.sf(-np.abs(z_statistics))))
    print("========================================================================")

    print("Hypothesis Testing - H_0 : p <= {} vs H_A : p > {}".format(p_0, p_0))

    z_statistics = (x - 0.5 - n*p_0)/sqrt(n*p_0*(1-p_0))
    print("z-statistics : {:.4f}".format(z_statistics))
    print("p-value = P(Z >= z) : {:.3f}".format(norm.sf(-np.abs(z_statistics))))
    print("========================================================================")

def compare_two_population_proportions(x, n, y, m, alpha):
    p_hat_A = x/n
    p_hat_B = y/m

    print("p_hat_A : {:.3f}, p_hat_B : {:.3f}".format(p_hat_A, p_hat_B))
    print("========================================================================")

    std_err = sqrt(p_hat_A*(1-p_hat_A)/n + p_hat_B*(1-p_hat_B)/m)
    print("C.I. of p_A - p_B with alpha {}".format(alpha))
    z_alpha = z_alpha_(1-(alpha/2))
    print("Two-Sided : ({:.3f}, {:.3f})".format(p_hat_A-p_hat_B-z_alpha*std_err, p_hat_A-p_hat_B+z_alpha*std_err))
    z_alpha = z_alpha_(1-alpha)
    print("Lower : ({:.3f}, 1)".format(p_hat_A-p_hat_B-z_alpha*std_err))
    print("Upper : (-1, {:.3f})".format(p_hat_A-p_hat_B+z_alpha*std_err))
    print("========================================================================")

    print("Hypothesis Testing - H_0 : p_A = p_B vs H_A : p_A ≠ p_B")
    p_hat = (x+y)/(n+m)
    z_statistics = (p_hat_A-p_hat_B)/sqrt(p_hat*(1-p_hat)*(1/n+1/m))
    print("z-statistics : {:.4f}".format(z_statistics))
    print("p-value = 2P(Z <= -|z|) : {:.3f}".format(2*(1-norm.sf(-np.abs(z_statistics)))))
    print("========================================================================")

    print("Hypothesis Testing - H_0 : p_A >= p_B vs H_A : p_A < p_B")
    print("z-statistics : {:.4f}".format(z_statistics))
    print("p-value = P(Z <= z) : {:.3f}".format((norm.sf(-np.abs(z_statistics)))))
    print("========================================================================")

    print("Hypothesis Testing - H_0 : p_A <= p_B vs H_A : p_A > p_B")
    print("z-statistics : {:.4f}".format(z_statistics))
    print("p-value = P(Z >= z) : {:.3f}".format(1-norm.sf(-np.abs(z_statistics))))
    print("========================================================================")

def one_way_contigency_table(x, p):
    """
    x = [x1, x2, ... , xk]
    p = [p1*, p2*, ... , pk*]
    """

    x, p = np.array(x), np.array(p)
    n = np.sum(x)
    e = np.multiply(n, p)
    
    # Person's Chi-Square statistic
    X_square = np.sum(((x-e)**2)/e)
    
    # Likelihood ratio Chi-Square statistic
    G_square = 2*np.sum(x*np.log(x/e))

    print("Fit Tests for One-Way Contigency Table with")
    print("x = {}".format(x))
    print("p* = {}".format(p))
    print("expected cell frequencies : {}".format(e))
    print("========================================================================")

    print("H_0 : p_i = p_i* for all i vs H_A : not H_0")
    print("X^2 = {:.4f}, G^2 = {:.4f}".format(X_square, G_square))
    print("p-value : P(X^2 >= {:.4f}) = {:.6f}, P(X^2 >= {:.4f}) = {:.6f}".format(X_square, 1-chi2.cdf(X_square, len(x)-1), G_square, 1-chi2.cdf(G_square, len(x)-1)))
    
def two_way_contigency_table(x):
    """
    x = [[x11, x12, ... , x1c],
         [x21, x22, ... , x2c],
                  
                  ...

         [xr1, xr2, ... , xrc]]
    """

    x = np.array(x)
    n = np.sum(x)
    row_marginal = np.sum(x, axis=1) # [x1., x2., ... , xr.]
    col_marginal = np.sum(x, axis=0) # [x.1, x.2, ... , x.c]

    df = (len(row_marginal)-1)*(len(col_marginal)-1)
    e = np.outer(row_marginal, col_marginal)/n

    # Person's Chi-Square statistic
    X_square = np.sum(((x-e)**2)/e)

    # Likelihood ratio Chi-Square statistic
    G_square = 2*np.sum(x*np.log(x/e))

    print("Fit Tests for Two-Way Contigency Table with")
    print("x = {}".format(x))
    print("expected cell frequencies : {}".format(e))
    print("[x1., ... , xr.] = {}".format(row_marginal))
    print("[x.1, ... , x.c] = {}".format(col_marginal))
    print("========================================================================")

    print("H_0 : Two factors are indep. vs H_A : not H_0")
    print("X^2 = {:.4f}, G^2 = {:.4f}".format(X_square, G_square))
    print("p-value : P(X^2 >= {:.4f}) = {:.6f}, P(X^2 >= {:.4f}) = {:.6f}".format(X_square, 1-chi2.cdf(X_square, df), G_square, 1-chi2.cdf(G_square, df)))

"""
Chap 11. The Analysis of Variance
"""
def ANOVA(n_T, k, SSTr, SSE):
    df1 = k-1
    df2 = n_T-k
    
    MSTr = SSTr/(k-1)
    MSE = SSE/(n_T-k)

    F_statistic = MSTr/MSE
    p_value = f.sf(F_statistic, df1, df2)

    print("ANOVA with n_T {}, k {}, SSTr {}, SSE {}".format(n_T, k, SSTr, SSE))
    print("MSTr : {:.4f}, MSE : {:.4f}".format(MSTr, MSE))
    print("F-statistic : {:.4f}, p-value = P(F_(k-1,n_T-k) >= {:.4f}) = {:.6f}".format(F_statistic, F_statistic, p_value))

def pairwise_CI(x1_var, x2_var, n1, n2, nT, k, MSE, alpha):
    print("1-α confidence level simultaneous confidence intervals for difference")
    print("with x1_var {}, x2_var {}, n1 {}, n2 {}, nT {}, k {}, MSE {}, α {}".format(x1_var, x2_var, n1, n2, nT, k, MSE, alpha))

    q = float(input("Please enter q_({},{},{}) from the Table : ".format(alpha, k, nT-k)))
    print("({:.4f}, {:.4f})".format(x1_var-x2_var-q*sqrt((MSE/2)*(1/n1 + 1/n2)), x1_var-x2_var+q*sqrt((MSE/2)*(1/n1 + 1/n2))))

"""
Chap 12. Simple Linear Regression and Correlation
"""



def main():
    opt = parse_option()

if __name__ == '__main__':
    main()
    # t_intervals(49.999, 0.134, 60, 0.1, 0.01)
    # z_intervals(49.999, 0.134, 60, 0.1, 0.01)
    # z_tests_on_mean(50.0, 49.99856, 0.1334, 60, 0.1)
    # two_indep_sample_problem(24, 34, 9.005, 11.864, 3.438, 3.305, 0.05, 0, 'z')
    # sample_proportion(1250, 98, 0.1, 0.01)
    # compare_two_population_proportions(406, 6000, 83, 2000, 0.01)
    # one_way_contigency_table([9, 24, 13], [.2, .5, .3])
    # two_way_contigency_table([[122,30,20,472],
    #                           [226, 51, 66, 704],
    #                           [306, 115, 96, 1072],
    #                           [130, 59, 38, 501],
    #                           [50, 31, 15, 249]])
    # ANOVA(35, 3, 204, 138.5)
    pairwise_CI(11.209, 15.086, 11, 14, 35, 3, 4.33, 0.05)