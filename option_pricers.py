import math
from scipy.stats import norm
import matplotlib.pyplot as plt


class European:
    def __init__(self, s0, k, ttm, r, vol, type):
        self.s0 = s0
        self.k = k
        self.ttm = ttm
        self.r = r
        self.vol = vol
        self.type = type

    def __repr__(self):
        return f'European {self.type} option. S0:{self.s0}  K:{self.k}  TTM:{self.ttm}  VOL{self.vol}  RF:{self.r}'
        pass

    def bsm_price(self):
        d1 = (math.log(self.s0/self.k) + self.ttm * (self.r + self.vol ** 2 / 2)) \
             / self.vol / math.sqrt(self.ttm)

        d2 = d1 - self.vol * math.sqrt(self.ttm)

        if self.type.lower() == 'call':
            price = self.s0 * norm.cdf(d1) - self.k * math.exp(-self.r * self.ttm) * norm.cdf(d2)
        elif self.type.lower() == 'put':
            price = self.k * math.exp(-self.r * self.ttm) * norm.cdf(-d2) - self.s0 * norm.cdf(-d1)
        else:
            raise TypeError('type must be call or put')

        return round(price, 2)


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
        return round(option_price, 3)

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
        return round(option_price, 3)


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
        return None

if __name__ == '__main__':
    #test1 = European(s0=100, k=110, ttm=0.5, r=0.04, vol=0.3, type='call')
    #print(f'European Call Price: {test1.bsm_price()}')

    #test1 = European(s0=100, k=110, ttm=0.5, r=0.04, vol=0.3, type='Put')
    #print(f'European Put Price: {test1.bsm_price()}')

    test2 = American(s0=50, k=50, ttm=0.4167, r=0.1, vol=0.4, type='Put')
    print(f'American Put Price, Binomial Tree: {test2.bin_tree_price(steps=50)}')

    #test2 = American(s0=50, k=50, ttm=0.4167, r=0.1, vol=0.4, type='call')
    #print(f'American Call Price, Binomial Tree: {test2.bin_tree_price(steps=5)}')

    test2.bin_tree_convergence(50)

    print(f'American Put Price, Trinomial Tree: {test2.tri_tree_price(steps=50)}')
    test2.tri_tree_convergence(50)

