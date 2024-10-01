A repo for demonstrating diffusion-based deterministic hyper-RL-policies. We would like to optimize a sampling distribution over deterministic optimal policies. 

Toy setup:
> Monitoring example in https://arxiv.org/abs/2102.11941.

First idea:
> Condition an (oracle) policy network with (state-augmenting) dual multipliers. For each given dual multiplier, the policy network outputs a (Lagrangian minimizing) deterministic policy. The dual multiplier sampling distribution is optimized either in online or offline manner. ---> Stochastic optimal control.

![alt text](https://github.com/yigit-uslu/Diffusion-HyperPolicies/blob/master/figures/monitoring-hyperpolicy.jpg?raw=true)
