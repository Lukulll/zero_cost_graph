from model.NAS_Bench_Graph_model import Net,get_NAS_Bench_graph_HP,get_num_features,get_num_classes
from pruners.predictive import find_measures
from utils.weight_initializers import init_net

if __name__ == '__main__':
  operations = ["gat", "gcn", "gin", "cheb"]
  link = [0,1,2,3]
  dname = 'CS'
  in_dim = get_num_features(dname)
  out_dim = get_num_classes(dname)
  device = 'cpu'
  hp = get_NAS_Bench_graph_HP()
  model = Net(operations, link, hp, in_dim, out_dim, dname).to(device)
  init_net(model,'kaiming','zero')
  measure = 'synflow'
  
  print(find_measures(model,dname,device, measure_names = [measure]))

