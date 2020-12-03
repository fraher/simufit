import simufit as sf
from simufit import DistributionType as dt
import numpy as np

class Tests():

    def __init__(self):
        self._seed = 12345
        self._weibull = 0.0
        self._bernoulli = 0.0
        self._binomial = 0.0
        self._geometric = 0.0
        self._uniform = 0.0
        self._normal = 0.0
        self._exponential = 0.0
        self._gamma = 0.0
        self._samples_min = 1000
        self._samples_max = 10000

    def printReport(self):
        print('Weibull: ', self._weibull)
        print('Bernoulli: ', self._bernoulli)
        print('Binomial: ', self._binomial)
        print('Geometric: ', self._geometric)
        print('Uniform: ', self._uniform)
        print('Normal: ', self._normal)
        print('Exponential: ', self._exponential)
        print('Gamma: ', self._gamma)

    def runTests(self):
        self.weibull()        
        self.bernoulli()
        self.binomial()
        self.geometric()
        self.uniform()
        self.normal()
        self.exponential()
        self.gamma()


    def bernoulli(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,300):
            test_count +=1

            p=np.random.random()
            size=np.random.randint(self._samples_min,self._samples_max)
            x = sf.Distribution()
            x.setDistribution(dt.UNKNOWN)        

            y = sf.Distribution()
            y.setDistribution(dt.BERNOULLI)
            y.generateSamples(p=np.random.random(), size=size)

            x.setSamples(y.getSamples())            
            x.identifyDistribution(n=5)
                    
            if x.Distribution.name.title() == dt.BERNOULLI.name.title():
                test_success += 1

        self._bernoulli = test_success/test_count

    def binomial(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,50):        
            for n in [2, 5, 7, 10, 15, 20]:
                test_count +=1
                
                size=np.random.randint(self._samples_min,self._samples_max)
                x = sf.Distribution()
                x.setDistribution(dt.UNKNOWN)        

                y = sf.Distribution()
                y.setDistribution(dt.BINOMIAL)
                y.generateSamples(n=n, p=np.random.random(), size=size)

                x.setSamples(y.getSamples())            
                x.identifyDistribution(n=n)
                        
                if x.Distribution.name.title() == dt.BINOMIAL.name.title():
                    test_success += 1

        self._binomial = test_success/test_count

    def geometric(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,300):
            test_count +=1

            p=np.random.random()
            size=np.random.randint(self._samples_min,self._samples_max)
            x = sf.Distribution()
            x.setDistribution(dt.UNKNOWN)        

            y = sf.Distribution()
            y.setDistribution(dt.GEOMETRIC)
            y.generateSamples(p=np.random.random(), size=size)

            x.setSamples(y.getSamples())            
            x.identifyDistribution(n=5)
                    
            if x.Distribution.name.title() == dt.GEOMETRIC.name.title():
                test_success += 1

        self._geometric = test_success/test_count

    def uniform(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,300):
            test_count +=1

            a = np.random.randint(1,49)
            b = np.random.randint(50,100)
            size=np.random.randint(self._samples_min,self._samples_max)
            x = sf.Distribution()
            x.setDistribution(dt.UNKNOWN)        

            y = sf.Distribution()
            y.setDistribution(dt.UNIFORM)
            y.generateSamples(a=a, b=b, size=size)

            x.setSamples(y.getSamples())            
            x.identifyDistribution(n=5)
                    
            if x.Distribution.name.title() == dt.UNIFORM.name.title():
                test_success += 1

        self._uniform = test_success/test_count

    def normal(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,300):
            test_count +=1

            distribution = np.random.randint(1,100,size=100)
            mean = np.average(distribution)
            var = np.var(distribution)

            size=np.random.randint(self._samples_min,self._samples_max)
            x = sf.Distribution()
            x.setDistribution(dt.UNKNOWN)        

            y = sf.Distribution()
            y.setDistribution(dt.NORMAL)
            y.generateSamples(mean=mean, var=var, size=size)

            x.setSamples(y.getSamples())            
            x.identifyDistribution(n=5)
                    
            if x.Distribution.name.title() == dt.NORMAL.name.title():
                test_success += 1

        self._normal = test_success/test_count

    def exponential(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,300):
            test_count +=1

            lambd = np.random.randint(0,10)+np.random.random()            
            size=np.random.randint(self._samples_min,self._samples_max)
            x = sf.Distribution()
            x.setDistribution(dt.UNKNOWN)        

            y = sf.Distribution()
            y.setDistribution(dt.EXPONENTIAL)
            y.generateSamples(lambd=lambd, size=size)

            x.setSamples(y.getSamples())            
            x.identifyDistribution(n=5)
                    
            if x.Distribution.name.title() == dt.EXPONENTIAL.name.title():
                test_success += 1

        self._exponential = test_success/test_count

    def gamma(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,300):
            test_count +=1

            a = np.random.randint(0,100)+np.random.random()            
            size=np.random.randint(self._samples_min,self._samples_max)
            x = sf.Distribution()
            x.setDistribution(dt.UNKNOWN)        

            y = sf.Distribution()
            y.setDistribution(dt.GAMMA)
            y.generateSamples(a=a, size=size)

            x.setSamples(y.getSamples())            
            x.identifyDistribution(n=5)
                    
            if x.Distribution.name.title() == dt.GAMMA.name.title():
                test_success += 1

        self._gamma = test_success/test_count

    def weibull(self):
        np.random.seed(self._seed)

        test_count = 0
        test_success = 0

        for count in range(1,300):
            test_count +=1

            a=np.random.randint(3,10)+np.random.random()
            b=np.random.randint(3,10)
            size=np.random.randint(self._samples_min,self._samples_max)
            x = sf.Distribution()
            x.setDistribution(dt.UNKNOWN)        

            y = sf.Distribution()
            y.setDistribution(dt.WEIBULL)
            y.generateSamples(a=a, b=b, size=size)

            x.setSamples(y.getSamples())            
            x.identifyDistribution(n=5)

            if x.Distribution.name.title() == dt.WEIBULL.name.title():
                test_success += 1

        self._weibull = test_success/test_count