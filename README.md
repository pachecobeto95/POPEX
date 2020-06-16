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

The component Decision Mkaer is responsible to select, dinamically, the optimal partitioning layer, using the parameters provided by the Background. Therefore, Decision Maker is able to determine which BranchyNet layers are processed at the edge device or at the cloud server. Decision maker is divided in four tasks: (1) estimation of inference time in BranchyNet; (2) optimization problem; (3) generation of multiples partitioning strategies; (4) decision of partitioning strategy. It is imporante to notice that, the first three tasks are executally, periodically, while the fourth is executed continually in inference process, in other words, it is executed whenever edge receives receives an input data. 

\begin{figure}[!htp]
    \centering
    \includegraphics[width=\linewidth]{imgs_cap6/decision_maker2.pdf}
    \caption{Ilustrações detalhada das tarefas que compõe o \textit{Decision Maker}}
    \label{fig:decision_maker_architecture}
\end{figure}

No primeiro momento, o \textit{Decision Maker} recebe os parâmetros de tempo de processamento na borda e na nuvem, além do grafo associado a arquitetura da BranchyNet e o tempo de comunicação atual vindos do componente \textit{Background}. 
Em seguida, o \textit{Decision Maker} executa a etapa de estimação do tempo de inferência. Em razão da BrachyNet possibilitar que amostras sejam classificadas antecipadamente nas camadas intermediárias, o tempo de inferência também está relacionado a probabilidade de classificar nos ramos laterais. Portanto, a modelagem do tempo de inferência deve descrever essas especifidades da BranchyNet. 

Posteriormente, o \textit{Decision Maker} executa a otimização que encontra a estratégia de particionamento ótimo que minimize o tempo de inferência. 
Contudo, conforme apresentado na Seção~\ref{sec:branchyNet}, a decisão do particionamento ótimo depende também do limiar de confiança, o que afeta o tempo de inferência, a acurácia, e consequentemente a decisão de particionamento. Logo, toda decisão de particionamento ótima está associada a uma configuração do limiar de entropia. Consequentemente, cada configuração dos limiares de entropia gera diferentes estratégias de particionamento ótimo. 
Portanto, é necessário escolher a configuração de limiar de entropia que equilibre o compromisso entre acurácia e tempo de inferência. Para isso, executa-se a geração de múltiplas estratégias de particionamento. 
Por fim, o \textit{Decision Maker} executa a etapa de decisão, a qual tem como objetivo selecionar aquela que maximiza a acurácia e atende ao requisito do tempo de inferência máximo informado pela aplicação.  

A seção a seguir formaliza o problema e particionamento de BranchyNet, em seguida, as próximas seções detalham cada etapas do problema de otimização, a geração de múltiplas estratégias de particionamento e, finamente, a etapa de decisão.


