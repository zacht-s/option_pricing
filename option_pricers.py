import math
import time
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class European:
    def __init__(self, s0, k, ttm, r, vol, type):
        self.s0 = s0
        self.k = k
        self.ttm = ttm
        self.r = r
        self.vol = vol
        self.type = type

        if self.type.lower() not in ['call', 'put']:
            raise TypeError('Type Variable must be Call or Put')

    def __repr__(self):
        return f'European {self.type} option. S0:{self.s0}  K:{self.k}  TTM:{self.ttm}  VOL{self.vol}  RF:{self.r}'
        pass

    def bsm_price(self):
        d1 = (math.log(self.s0/self.k) + self.ttm * (self.r + self.vol ** 2 / 2)) \
             / self.vol / math.sqrt(self.ttm)

        d2 = d1 - self.vol * math.sqrt(self.ttm)

        if self.type.lower() == 'call':
            price = self.s0 * norm.cdf(d1) - self.k * math.exp(-self.r * self.ttm) * norm.cdf(d2)
        else:
            price = self.k * math.exp(-self.r * self.ttm) * norm.cdf(-d2) - self.s0 * norm.cdf(-d1)

        return round(price, 2)

    def monte_carlo_price(self, sims):
        terminal_prices = self.s0 * np.exp((self.r - self.vol ** 2 / 2) * self.ttm +
                                           self.vol * math.sqrt(self.ttm) * np.random.normal(loc=0, scale=1, size=sims))

        if self.type.lower() == 'call':
            option_value = terminal_prices - self.k
        else:
            option_value = self.k - terminal_prices
        option_value = np.where(option_value < 0, 0, option_value)

        return round(np.mean(option_value) * math.exp(-self.r * self.ttm), 2)


class American:
    def __init__(self, s0, k, ttm, r, vol, type):
        self.s0 = s0
        self.k = k
        self.ttm = ttm
        self.r = r
        self.vol = vol
        self.type = type

        if self.type.lower() not in ['call', 'put']:
            raise TypeError('Type Variable must be Call or Put')

    def __repr__(self):
        return f'American {self.type} option. S0:{self.s0}  K:{self.k}  TTM:{self.ttm}  VOL{self.vol}  RF:{self.r}'
        pass

    def bin_tree_price(self, steps):
        dt = self.ttm/steps
        u = math.exp(self.vol * math.sqrt(dt))
        d = math.exp(-self.vol * math.sqrt(dt))
        p = (math.exp(self.r * dt) - d) / (u - d)

        def tree_node(s0, up_exp, dn_exp):
            # Helper Function
            node = [round(s0 * (u ** up) * (d ** dn), 4) for up, dn in zip(up_exp, dn_exp)]
            return node

        # Build Binomial Tree of Asset Prices
        asset_tree = [self.s0]
        for i in range(steps):
            up_pwr = list(range(i+2))
            dn_pwr = up_pwr[::-1]
            node = tree_node(s0=self.s0, up_exp=up_pwr, dn_exp=dn_pwr)
            asset_tree.append(node)

        # Work Backward through Binomial Tree to Calculate Option Price
        discount = math.exp(-self.r * self.ttm / steps)

        np1_prices = []
        for i in range(len(asset_tree), 1, -1):
            if self.type.lower() == 'call':
                imm_exercise = [max(x - self.k, 0) for x in asset_tree[i-1]]
            else:
                imm_exercise = [max(self.k - x, 0) for x in asset_tree[i-1]]

            if len(asset_tree[i-1]) == len(asset_tree):
                hold_vec = [0]*len(asset_tree)
            else:
                hold_vec = [(p * np1_prices[x_up] + (1 - p) * np1_prices[x_dn]) * discount
                            for x_dn, x_up in zip(range(len(np1_prices) - 1), range(1, len(np1_prices)))]

            np1_prices = [max(x, y) for x, y in zip(hold_vec, imm_exercise)]

        option_price = ((1 - p) * np1_prices[0] + p * np1_prices[1]) * discount
        return round(option_price, 2)

    def bin_tree_convergence(self, steps_max):
        """
        Graphical representation of the convergence of the binomial tree pricing as a function of the number of steps
        :param steps_max: Int, the max number of steps to try in the loop
        """
        prices = []
        steps = []
        for i in range(steps_max):
            prices.append(self.bin_tree_price(steps=i+1))
            steps.append(i+1)

        plt.plot(steps, prices, '-o')
        plt.suptitle(f'Convergence of American {self.type} Option Price via Binomial Tree')
        plt.title(f'S0: {self.s0}, K: {self.k}, TTM: {self.ttm}, VOL: {self.vol}, RF: {self.r}')
        plt.xlabel('Number of Steps')
        plt.ylabel('Option Price')
        plt.show()
        return None

    def tri_tree_price(self, steps):
        dt = self.ttm / steps
        u = math.exp(self.vol * math.sqrt(3*dt))
        pd = -math.sqrt(dt/12/self.vol**2) * (self.r - self.vol**2 / 2) + 1/6
        pm = 2/3
        pu = math.sqrt(dt/12/self.vol**2) * (self.r - self.vol**2 / 2) + 1/6

        # Build Trinomial Tree of Asset Prices
        asset_tree = [self.s0]
        prior = [0]
        for i in range(steps):
            pwrs = list(range(prior[0]-1, prior[-1]+2))
            prior = pwrs
            node = [round(self.s0 * u ** x, 4) for x in pwrs]
            asset_tree.append(node)

        # Work Backward through Trinomial Tree to Calculate Option Price
        discount = math.exp(-self.r * self.ttm / steps)

        np1_prices = []
        for i in range(len(asset_tree), 1, -1):
            if self.type.lower() == 'call':
                imm_exercise = [max(x - self.k, 0) for x in asset_tree[i - 1]]
            else:
                imm_exercise = [max(self.k - x, 0) for x in asset_tree[i - 1]]

            if len(asset_tree[i-1]) == len(asset_tree)*2 - 1:
                hold_vec = [0]*(len(asset_tree)*2 - 1)
            else:
                hold_vec = [(pd * np1_prices[x_dn] + pm * np1_prices[x_m] + pu * np1_prices[x_up]) * discount
                            for x_dn, x_m, x_up in zip(range(len(np1_prices)-2), range(1, len(np1_prices)-1),
                                                       range(2, len(np1_prices)))]

            np1_prices = [max(x, y) for x, y in zip(hold_vec, imm_exercise)]

        option_price = (pd * np1_prices[0] + pm * np1_prices[1] + pu * np1_prices[2]) * discount
        return round(option_price, 2)

    def tri_tree_convergence(self, steps_max):
        """
        Graphical representation of the convergence of the trinomial tree pricing as a function of the number of steps
        :param steps_max: Int, the max number of steps to try in the loop
        """
        prices = []
        steps = []
        for i in range(steps_max):
            prices.append(self.tri_tree_price(steps=i+1))
            steps.append(i+1)

        plt.plot(steps, prices, '-o')
        plt.suptitle(f'Convergence of American {self.type} Option Price via Trinomial Tree')
        plt.title(f'S0: {self.s0}, K: {self.k}, TTM: {self.ttm}, VOL: {self.vol}, RF: {self.r}')
        plt.xlabel('Number of Steps')
        plt.ylabel('Option Price')
        plt.show()
        return

    def monte_carlo_price(self, sims, steps, order=2):
        """
        Prices American Options via Monte Carlo Simulation. Value of holding the option at each timestep is estimated
        using a polynomial regression, per the Longstaff Schwartz Algorithm.
        :param sims: Int, Number of Simulations to Run
        :param steps: Int, Number of steps in each realization of the underlying price pathway
        :param order: Int, degree of the polynomial to use in regression, default of 2.
        :return: option price
        """

        dt = self.ttm / steps

        random_walks = np.exp((self.r - self.vol ** 2 / 2) * dt + \
                       self.vol * math.sqrt(dt) * np.random.normal(loc=0, scale=1, size=(sims, steps)))

        paths = np.cumprod(random_walks, axis=1) * self.s0

        colnames = []
        for i in range(steps):
            colnames.append(f't{i+1}')

        # Define Dataframes for Underlying price realizations, immediate execution values,
        # and Cash from Exercising the option at each timestep
        asset_df = pd.DataFrame(paths, columns=colnames)

        if self.type.lower() == 'call':
            imm_exc = asset_df - self.k
        else:
            imm_exc = self.k - asset_df
        imm_exc = imm_exc.mask(imm_exc < 0, 0)

        cash_df = imm_exc.copy()
        cash_df[:] = 0
        cash_df[imm_exc.columns[-1]] = imm_exc[imm_exc.columns[-1]]

        # Begin Backward Calculation of Option Price
        future_days = []
        disc_vec = np.array([np.exp(-self.r * dt)])
        for i in range(len(asset_df.columns)-1, 0, -1):
            future_days.append(f't{i + 1}')

            # Calculate the Expected Continuation Value E[v(t) | s(t-1)] by LongStaff Schwartz Regression
            Y = cash_df.loc[imm_exc[f't{i}'] > 0, future_days] * disc_vec
            Y = Y.sum(axis=1)
            X = asset_df.loc[imm_exc[f't{i}'] > 0, f't{i}']
            reg = np.polyfit(X, Y, order)
            cont_val = np.polyval(reg, X)

            # If Immediate Execution Value > Continuation Value, then option is exercised and later cash flows are 0
            update_vec = np.where(imm_exc.loc[imm_exc[f't{i}'] > 0, f't{i}'] < cont_val, 0,
                                  imm_exc.loc[imm_exc[f't{i}'] > 0, f't{i}'])

            cash_df.loc[imm_exc[f't{i}'] > 0, f't{i}'] = update_vec
            cash_df.loc[cash_df[f't{i}'] > 0, future_days] = 0

            disc_vec *= np.exp(-self.r * dt)
            disc_vec = np.append(disc_vec, np.exp(-self.r * dt))

        dcf = cash_df * disc_vec[::-1]
        option_value = round(dcf.sum(axis=0).sum() / len(asset_df.index), 2)
        return option_value

    def implicit_fd_price(self, t_steps, s_steps, s_max=None):
        dt = self.ttm / t_steps
        if s_max is None:
            ds = self.s0 * 2 / s_steps
        else:
            ds = s_max / s_steps

        def a_j(j):
            return 1/2 * self.r * j * dt - 1/2 * self.vol**2 * j**2 * dt

        def b_j(j):
            return 1 + self.vol**2 * j**2 * dt + self.r * dt

        def c_j(j):
            return -1/2 * self.r * j * dt - 1/2 * self.vol**2 * j**2 * dt

        df = pd.DataFrame(data=np.zeros(shape=(s_steps+1, t_steps+1)), index=range(s_steps, -1, -1))
        df.loc[:, t_steps] = np.maximum(self.k - np.arange(start=s_steps, stop=-1, step=-1) * ds, 0)
        df.loc[0, :] = self.k
        df.loc[s_steps, :] = 0

        # Generate Coefficient Matrix to solve for interior grid points
        A = []
        temp = [b_j(s_steps-1), a_j(s_steps-1)] + [0]*(s_steps-3)
        A.append(temp)
        for i in range(s_steps-2, 1, -1):
            temp = [0]*(s_steps-2-i)
            temp.append(c_j(i))
            temp.append(b_j(i))
            temp.append(a_j(i))
            temp += [0] * (i-2)
            A.append(temp)

        temp = [0]*(s_steps-3) + [c_j(1), b_j(1)]
        A.append(temp)
        A = pd.DataFrame(A).to_numpy()

        s_vec = np.arange(start=s_steps, stop=-1, step=-1) * ds

        for t_i in range(t_steps-1, -1, -1):
            # b is next timestep answer to Ax = b Matrix Equation
            b = df[t_i + 1].copy()
            b[s_steps - 1] = b[s_steps - 1] - c_j(s_steps - 1) * b[s_steps]
            b[1] = b[1] - a_j(1) * b[0]
            b = b.drop([0, s_steps])

            x = np.matmul(np.linalg.inv(A), b)

            # x is approximate solution to BSM, update is max(x, K-S) for immediate exercise option with
            # American style options
            temp = df.loc[:, t_i].copy()
            temp.loc[range(s_steps-1, 0, -1)] = x
            update_vec = np.maximum(df[t_steps], temp)
            df.loc[:, t_i] = update_vec

        # Numpy Interpolation Expects X (S0) values in increasing order
        option_price = np.interp(self.s0, s_vec[::-1], df[0].to_numpy()[::-1])
        return round(option_price, 2)


if __name__ == '__main__':
    """
    # European Option Testing Block
    test1 = European(s0=100, k=110, ttm=0.5, r=0.04, vol=0.3, type='call')
    print(f'European Call Price: {test1.bsm_price()}')
    print(f'Monte Carlo Call Price (1,000,000 Trials): {test1.monte_carlo_price(1000000)}')

    test1 = European(s0=100, k=110, ttm=0.5, r=0.04, vol=0.3, type='Put')
    print(f'European Put Price: {test1.bsm_price()}')
    print(f'Monte Carlo Put Price (1,000,000 Trials): {test1.monte_carlo_price(1000000)}')
    """

    """
    # American Option Testing Block
    test2 = American(s0=50, k=50, ttm=0.4167, r=0.1, vol=0.4, type='Put')
    n_steps = 100
    start = time.time()
    option_val = test2.bin_tree_price(steps=n_steps)
    end = time.time()
    print(f'American Put Price, Binomial Tree: {option_val}, Execution Time: {round(end-start, 2)} seconds')

    start = time.time()
    option_val = test2.tri_tree_price(steps=n_steps)
    end = time.time()
    print(f'American Put Price, Trinomial Tree: {option_val}, Execution Time: {round(end - start, 2)} seconds')

    n_sims = 100000
    start = time.time()
    option_val = test2.monte_carlo_price(n_sims, n_steps)
    end = time.time()
    print(f'American Put Price, LSM MonteCarlo: {option_val}, Execution Time: {round(end-start, 2)} seconds')

    tsteps, ssteps = 100, 100
    start = time.time()
    option_val = test2.implicit_fd_price(t_steps=tsteps, s_steps=ssteps)
    end = time.time()
    print(f'American Put Price, Implicit Finite Difference: {option_val}, '
          f'Execution Time: {round(end - start, 2)} seconds')
    print('')

    # test2.bin_tree_convergence(50)
    # test2.tri_tree_convergence(50)
    """

    pass
