## Two Agent Newsvendor Problem

### Description

Consider a logistics manager for Amazon, who needs to provide trailers to move freight out of Chicago on a weekly basis.
There is data to estimate how many trailers he will need for the week but this esimate is not certain.
Using this estimate, the manager then submits a request to the regional manager who will judges how many
trailers to provide to the local manager.

Despite working for the same company, each has their own concerns to focus on. The local, logististics manager is 
concerned about running out of trailers due to the extra of renting short-term trailers if there are not enough
trailers. Meanwhile, the regional manager does not want the local manager to run out while also not wanting
to provide any excess, due the increased cost covered by them. 

The process is as follows:
1. Local manager observes the initial estimate of how many trailers needed. This is private
to only the local manager.
   
2. Local manager submits a request for the number of trailers to the regional manager.

3. Regional manager then decides how many trailers to given agent *q*.

4. Local manager receives the number of trailers granted by the central manager, and then observes the
realized (actual) need for trailers.
   
5. Local and regional managers compute respective contributions.

### Basic Model

The problem will be modeled for both agents, since information is not the same for both. The local
manager will be referred to as *q*, and the regional manager will be referred to as *q'*.

#### State Variables

Let *R_{tq}^est* be the initial estimate of how many trailers needed, only available to the local manager. This is an 
estimate; therefore, we should account for any bias. Let *sigma_{tq}^est* be the bias of our initial estimate, calculated
as the difference between *R_{tq}^est*. 

Then, let *sigma_{tq}^regional* be the estimate how much the regional manager reduces the request by the local manager,
which we will soon define as *x_{tqq'}*. In a similar vein, the regional manager will attempt to learn the difference 
between request by the local manager and what they actually need. Thus, let *sigma_{tq'}^local* be the estimate of this
difference, where *x_{tqq'}* is the local manager requests number. 

Therefore, the state variable for the local manager is *S_{tq}=(R_{tq}^est, sigma_{tq}^est, sigma_{tq}^regional)*. Meanwhile,
the regional manager's state variable is *S_{tq'}=(x_{tqq'}, sigma_{tq'}^local)*.

#### Actions

Let *x_{tqq'}* be the number of trailers agent *q* asks from agent *q'*. Let *x_{tq'q}* be the number of trailers agent
*q'* gives to agent *q*, which is what gets implemented in real-life.

#### Exogenous Information

The main source of uncertainty is the initial estimate of the trailers needed. After we make some decision *x_{tqq'}*, 
we receive two types of information: what the regional manager grants us, and the realized demand. Let *x_{tq'q}* be the 
number of trailers the regional manager provides us. Let *\hat{R}_{t+1}* be the number of trailers the local manager
ends up needing in real-life (i.e., the realized demand). Thus, we can say the exogenous information for agent *q* is 
*W_{t+1,q}=(x_{tq'q}, \hat{R}_{t+1})*.

On the other hand, the only exogenous information for the regional manager is the realized demand. Thus, 
*W_{t+1,q'}=(\hat{R}_{t+1})*

#### Transitions

Only the sigma's are updated in this problem, **because it is an information acquisition problem**.  
