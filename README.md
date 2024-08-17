# Code for KDD-24 paper "Rethinking Order Dispatching in Online Ride-Hailing Platforms"

**Note**

Due to data management regulations, we are unable to disclose our company's private datasets. Therefore, we have replicated the GRC on a publicly available dataset of ride-hailing services in Chengdu from November 2016. Due to the lack of some data, such as pre- and post-ride cancellation rates and the probability of driver online/offline status, the GRC and simulator we replicated is a simplified version.

**Requirements**

+ python 3.8
+ torch 2.3.0
+ torch_geometric 2.5.3
+ geopy 2.4.1
+ pyscipopt 5.0.0


**Run**

+ Historical data collection

  + run KM matching:

  ```
  cd simulator
  python main.py
  ```

+ MDP table construction

  + run the MDP algorithm (KDD-18):

  ```
  cd algorithm
  python mdp.py
  ```

+ Offline dataset collection

  + deploy the MDP table:

  ```
  cd ..
  python eval.py mdp
  ```

+ Train GRC

  + train reward model:

  ```
  cd algorithm
  python reward_model.py
  ```

  + train env model:

  ```
  python env_model.py
  ```

  + train policy:

  ```
  python grc.py
  ```

+ Evaluation

  + deploy the GRC policy:

  ```
  cd ..
  python eval.py grc
  ```

  + a quick step to get the d-s gap figures:

  ```
  cd ..
  run draw.ipynb
  ```



