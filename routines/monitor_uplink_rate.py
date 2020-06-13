from pythonping import ping
import pandas as pd
import schedule
import time, os


def __monitorspeed__(destIP="8.8.8.8"):
	monitor_bandwidth_path = os.path.join("./appEdge", "api", "data_results", 
		"bandwidth_records.csv")

	response_list = ping(destIP, size=40, count=5)
	rtt_avg = response_list.rtt_avg_ms/2

	size_bits = 8*(48)

	
	uplink_rate = size_bits/(rtt_avg*10**(-6))

	uplink_rate_dict = {"bandwidth": [uplink_rate]}

	if os.path.exists(monitor_bandwidth_path):
		df = pd.read_csv(monitor_bandwidth_path)
		df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

	else:
		df = pd.DataFrame(columns=["bandwidth"])

	df = df.append(pd.DataFrame(uplink_rate_dict), ignore_index=True, sort=False)
	print("Inserting")
	df.to_csv(monitor_bandwidth_path)

def remove_bandwidth():
	monitor_bandwidth_path = os.path.join("./appEdge", "api", "data_results", 
		"bandwidth_records.csv")

	os.remove(monitor_bandwidth_path)



schedule.every(10).seconds.do(__monitorspeed__)
schedule.every(24).hours.do(remove_bandwidth)


while(1):
	schedule.run_pending()
	time.sleep(1)