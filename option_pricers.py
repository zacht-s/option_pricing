import math
from scipy.stats import norm


class European:
    def __init__(self, s0, k, ttm, r, vol, type):
        self.s0 = s0
        self.k = k
        self.ttm = ttm
        self.r = r
        self.vol = vol
        self.type = type

    def __repr__(self):
        return 'These are European Options'
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


if __name__ == '__main__':
    test1 = European(s0=100, k=110, ttm=0.5, r=0.04, vol=0.3, type='call')
    #print(f'European Call Price: {test1.bsm_price()}')

    test1 = European(s0=100, k=110, ttm=0.5, r=0.04, vol=0.3, type='Put')
    #print(f'European Put Price: {test1.bsm_price()}')






