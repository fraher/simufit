Instructions on how to use Distribution() in the simufit namespace:

import simufit as sf
from simufit import DistributionType as dt
x = sf.Distribution()
x.setDistribution(dt.GEOMETRIC)

x.display
... (Example output)
Seed:  None
Range:  None - None
Size:  None
Samples:  []
Continuous:  None
Discrete: None
Distribution:  DistributionType.GEOMETRIC
...

samples = x.Distribution.sample(p=0.5,size=10) # In the future we can directly set the _sample value directly
x.setSamples(samples)
x.fit() 
x.Distribution.MLE(x._samples, use_minimizer=True, x0=0.42)