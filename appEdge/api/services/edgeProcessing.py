import pandas as pd
from ..branchMe.networks.alex_conv import get_network


def extract_processing_time():
	x = torch.rand((1, 3, 224, 224))
	branchyNet = get_network()
	branchyNet.to_cpu()
	branch_models,_ = branchyNet.initiate_branch2()

	N_ROUNDS = 20
	dict_runtime_total = {}


	for k, model in enumerate(branchyNet.models, 1):
		for i in range(N_ROUNDS):
			save_path = "benchmark_processing_layers_edge_model_%s.csv"%(k)

			_, dict_runtime = model(x, k)
			if(i==0):
				df = pd.DataFrame(columns = list(dict_runtime.keys()))
			#print(df)
			#print(list(dict_runtime.values()))
			df.loc[i] = list(dict_runtime.values())
			df.to_csv(save_path)

			#df = df.append(dict_runtime, ignore_index=True)
			#dict_runtime_total.update(dict_runtime)

	return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
