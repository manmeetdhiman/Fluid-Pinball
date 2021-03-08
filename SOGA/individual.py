sim_type = 'Rastrigin' #choose Rastrigin or Pinball
#performance limits
size = 100
#GA Parameters
starting_gen = 0
abs_counter = 0
#sim_type('Rastrigin')
target_cost = 0
max_gen = 100000
mut_prob = 0.1
mut_type = 'all' #choose between 'all','motor','gene','n_genes'
search_limit = 50000000
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
        self.sensor = {
            'top': [],
            'middle': [],
            'bottom': []
        }
        #defining fluctuation cost
        self.j_total = 0
        self.j_fluc = 0
        self.j_act = 0
        self.ras = 0
        #defining motor revolutions
        self.revolutions = [[], [], []]