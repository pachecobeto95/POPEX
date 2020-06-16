# POPEX (Partitioning OPtimization Early eXit network)

This project proposes POPEX, a early exit networks partitioning system. In a broader view, POPEX aims to accelerate the inference time, reducing not only communication delay though DNN partitioning, but also reducing the processing delay using early exit networks (e.g. BranchyNet). Therefore, this project aims to allow latency-sensiivity applications, such as smart car and cognitive assistance. To this end, POPEX splits a early exit network in two parts, so that one is processed at the edge device (e.g. smartphone, wearables and laptops), while the other part is processed at the cloud server. 



# Architecture
This project is divided in two main component: Background and Decision Maker. The architecture of the system is shown below. 





The Background component is responsable for monitoring the uplink rate of the network infrastructure between the edge device and tje cloud server, and also for extracting the optimization parameters. The Decision Maker executes the adaptive partitioning, according the optimization parameters and uplink rate extracted by the Background component. The adaptive partitioning consists of selecting an optimal partitioning, that maximizes the classification accuracy while also achieves a pre-defined user latency requirement. 

First of all, the applicaion provide two informations to POPEX system: latency requirement and the employed BranchyNet architecture. The latter is inserted in Background component, while the former is inserted at the Decision Maker to execute adaptive partitioning. The latency requirement refers to the maximum inference time allowed to classify an input sample.
The edge node, in Figure above, corresponds to a generic edge device such as smartphone, wearable or even a access point.  


Each component of this project is execute is explained in detail throughout the next sections.

## Background

Once the Background has received latency requirement and BranchyNet architecture from application, the Backgorund component extracts the parameters parameters that is executed by the Decision Maker component. Background executes three tasks: extraction of static parameters, uplink rate monitoring and construction of a graph related to BranchyNet architecture provided by the application in edge node. In the first task, the Background extracts the processing time and output data size of each layer from BranchyNet. These parameters are specific to Branchynet architecture and processing power of edge device, thus these parameters are static, which means that it is necessary to be executed only once during the system boot.  

Besides that, the Background also monitors the uplink rate between edge device and cloud server. To this end, Background uses ping tool and storage the uplink rate. Then, Background can calculate the communication delay required to send output data of each layer, using uplink rate and output data size. Background monitors the uplink rate continuously, using ping tool and storage this data for a period of time. 


Finally, the third task consists of constructing a graph based on BranchyNet architecture, where each layer corresponds to a vertex and a communication between layers represents the links in this graph. Therefore, POPEX convert a BranchyNet partitioning problem into a graph partitioning problem.  

\begin{figure}[!htp]
    \centering
    \includegraphics[width=0.8\linewidth]{imgs_cap5/background.pdf}
    \caption{Estrutura do componente \textit{Background}}
    \label{fig:background_component}
\end{figure}


## Decision Maker

The component Decision Maker is responsible to select, dinamically, the optimal partitioning layer, using the parameters provided by the Background. Therefore, Decision Maker is able to determine which BranchyNet layers are processed at the edge device or at the cloud server. Decision maker is divided in four tasks: (1) estimation of inference time in BranchyNet; (2) optimization problem; (3) generation of multiples partitioning strategies; (4) decision of partitioning strategy. It is imporante to notice that, the first three tasks are executally, periodically, while the fourth is executed continually in inference process, in other words, it is executed whenever edge receives receives an input data. 

\begin{figure}[!htp]
    \centering
    \includegraphics[width=\linewidth]{imgs_cap6/decision_maker2.pdf}
    \caption{Ilustrações detalhada das tarefas que compõe o \textit{Decision Maker}}
    \label{fig:decision_maker_architecture}
\end{figure}

At first, Decision Maker receives parameters of processing time at the edge and at cloud, in addition to the graph associated to BranchyNet architecture and the current communication time from Background. Then, Decision Makes constructs a new graph to convert the partitioning graph problem into a shortest path problem. Once constructed this new graph, we assign the weights in the links os this graph. As BranchyNet allows input samples to be classified at side branch, the weights assigned to the links are related to a probability of classifying a sample in a given side branch. This step in presented in detail [here](https://arxiv.org/pdf/2005.04099.pdf). At this stage, Decision Maker can execute the optimization problem to select the optimal partitioning that minimizes the inference time, using Dijkstra's algorithm. However, this particioning decision depends on several parameters, including the hyperparameters of the BranchyNet such as the threshold associated to each side branch. These threshold configuration decides whether an input sample can be classified at side branch or must be processed by the next layers. The choice of a threshold configurations handle to a trade-off between classification accuracy and inference time.  
For example, when the entropy threshold of the first side branch is set to high values, such as 0.9, the first side branch can classify poorly confident samples, decreasing accuracy and inference time. Otherwise, when the entropy threshold of this side branch is set to low values, such as 0.1, only high confident input data can be classified at the first side branch, increasing accuracy and also inference time since the majority of samples requires to be processed by the next layers. 
After that, we vary the entropy threshold of side branches and choose the partitioning strategy using the optimization method described in [here](https://arxiv.org/pdf/2005.04099.pdf), which minimizes the inference time. 
Each entropy threshold configuration results in a different partitioning strategy with a specific pair of classification accuracy and inference time associated. At this stage, for a given uplink rate, there are multiples partitioning strategy stored for each threshold configuration with a unique pair of accuracy and inference time
This step is also executed only once during system boot. At this point, finally, Decision Maker can excute the decision task, whoch goal is to select the partitioning strategy that maximizes the accuracy, while meets the pre-defined inference time provided by application. 


This repository containing the code to reproduce result found in "Inference Time Optimization Using BranchyNet Partitioning" paper. If you want to use this codebase, please cite:

    @article{pacheco2020inference,
        title={Inference Time Optimization Using BranchyNet Partitioning},
        author={Pacheco, Roberto G and Couto, Rodrigo S},
        journal={arXiv preprint arXiv:2005.04099},
        year={2020}
    }



## Deployment 
POPEX is deployed using a edge computing infrastructure. This project deploys a Web API on edge device and cloud server using the Flask framework. The edge node has an Web API to receive the BranchyNet architecture from application and to receive images from the end devices. When application provides the BranchyNet, POPEX can execute the parameters extraction. Then, the edge devices receives an image from end device, POPEX can select the partitioning decision based on extracted parameters by Background. Once the partitioning strategy is choosen, edge device send to the cloud the choosen partitioning layer and the output data from partitioning layer. 

## Requirements
* Python 3.0+
* pytorch
* matplotlib
* pandas
* scipy
* Flask







