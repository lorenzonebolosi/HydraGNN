class Data_object:

    def __init__(self, input_matrix):
        self.pos = input_matrix[:, :2]
        self.y = input_matrix[:, 2:4]
        self.x = input_matrix[:, 4:]
        self.edge_attr = None
        self.iterator = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.iterator += 1
        if(self.iterator == len(self.pos)):
            self.iterator = 0
            raise StopIteration
        return self.pos[self.iterator, :]
