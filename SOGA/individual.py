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