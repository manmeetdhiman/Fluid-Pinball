sim_type = 'Pinball' #choose Rastrigin or Pinball
#performance limits
size = 18
#GA Parameters
starting_gen = 0
#sim_type('Rastrigin')
target_cost = 0
max_gen = 30
mut_prob = 0.1
mut_type = 'all' #choose between 'all','motor','gene','n_genes'
search_limit = 50000000
gen_buffer_limit = 50 #number of generations GA is allowed to saturate
if mut_type == 'n_genes':
    n_genes = 7
else:
    n_genes = 1

#Simulation Parameters
dt = 5e-4
tsteps = 100

#defining individual class
#Motor order defined in front, top, bottom
class individual:
    def __init__(self):
        #defining genes
        self.id = 0
        self.genes = {
            'amplitude': [],
            'frequency': [],
            'phase': [],
            'offset': []
        }
        #defining fluctuation cost
        self.j_total = 0
        self.j_fluc = 0
        self.j_act = 0
        self.ras = 0
        #defining motor revolutions
        self.revolutions = [[], [], []]