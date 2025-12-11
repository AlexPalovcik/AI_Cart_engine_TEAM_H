from numpy import random as np_random
import random
import numpy as np
import copy
import string

# počet vstupů – ideálně = len(RAYCAST_ANGLES)
N_INPUTS = 9
N_ACTIONS = 4  # [up, down, left, right] - MUSÍ ZŮSTAT 4 kvůli kompatibilitě
N_HIDDEN_1 = 32 # První skrytá vrstva
N_HIDDEN_2 = 16 # Druhá skrytá vrstva


# vždy pojmenováváme jako "AIbrain_jemnoteamu"
class AIbrain_shallowH:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0
        self.x = 0
        self.y = 0
        self.speed = 0

        self.init_param()

    def init_param(self):
        """
        Inicializace vah a biasů pro 3 vrstvou síť (2 skryté vrstvy, Deep NN).
        Biasy se inicializují malými náhodnými hodnotami (prolomení symetrie).
        """
        rng = np_random.default_rng()
        
        # --- 1. vrstva (Vstup -> Skrytá 1) ---
        weight_scale_1 = 1.0 / np.sqrt(N_INPUTS)
        self.W1 = rng.normal(loc=0.0, scale=weight_scale_1,
                             size=(N_HIDDEN_1, N_INPUTS))
        # Inicializujeme biasy malými náhodnými hodnotami
        self.b1 = rng.normal(loc=0.0, scale=0.01, size=(N_HIDDEN_1,)) 

        # --- 2. vrstva (Skrytá 1 -> Skrytá 2) ---
        weight_scale_2 = 1.0 / np.sqrt(N_HIDDEN_1)
        self.W2 = rng.normal(loc=0.0, scale=weight_scale_2,
                             size=(N_HIDDEN_2, N_HIDDEN_1))
        # Inicializujeme biasy malými náhodnými hodnotami
        self.b2 = rng.normal(loc=0.0, scale=0.01, size=(N_HIDDEN_2,))
        
        # --- 3. vrstva (Skrytá 2 -> Výstup 4 akce) ---
        weight_scale_3 = 1.0 / np.sqrt(N_HIDDEN_2)
        self.W3 = rng.normal(loc=0.0, scale=weight_scale_3,
                             size=(N_ACTIONS, N_HIDDEN_2))
        # Inicializujeme biasy malými náhodnými hodnotami
        self.b3 = rng.normal(loc=0.0, scale=0.01, size=(N_ACTIONS,))


        self.NAME = "teamH_deep_4action"
        self.store()

    @staticmethod
    def _relu(z):
        """ Aktivační funkce ReLU. """
        return np.maximum(z, 0.0)

    def decide(self, data):
        """
        Propagace vpřed. Vrací 4 surové hodnoty pro 4 akce, 
        kompatibilní s původním systémem.
        """
        self.decider += 1

        x = np.asarray(data, dtype=float).ravel()
        
        # Ošetření velikosti vstupu
        n_w = N_INPUTS
        if x.size < n_w:
            x = np.concatenate([x, np.zeros(n_w - x.size)])
        elif x.size > n_w:
            x = x[:n_w]
        
        # ------------------------------------
        # 1. vrstva: Vstup -> Skrytá 1 (ReLU)
        z1 = self.W1.dot(x) + self.b1
        h1 = self._relu(z1)

        # 2. vrstva: Skrytá 1 -> Skrytá 2 (ReLU)
        z2 = self.W2.dot(h1) + self.b2
        h2 = self._relu(z2)

        # 3. vrstva: Skrytá 2 -> Výstup (Lineární aktivace)
        z3 = self.W3.dot(h2) + self.b3  # z3 má shape (N_ACTIONS=4,)
        # ------------------------------------

        # Vracíme 4 raw výstupy; AI_car aplikuje threshold > 0.5
        return z3

    def mutate(self):
        """
        Mutace se zvýšenou silou.
        """
        mutation_rate = 0.3 # Zvýšená síla mutace pro rychlejší únik z lokálních minim

        # --- Mutace pro W1 a b1 ---
        delta_W1 = (np_random.rand(*self.W1.shape) - 0.5) * mutation_rate
        delta_b1 = (np_random.rand(*self.b1.shape) - 0.5) * mutation_rate
        self.W1 = self.W1 + delta_W1
        self.b1 = self.b1 + delta_b1

        # --- Mutace pro W2 a b2 ---
        delta_W2 = (np_random.rand(*self.W2.shape) - 0.5) * mutation_rate
        delta_b2 = (np_random.rand(*self.b2.shape) - 0.5) * mutation_rate
        self.W2 = self.W2 + delta_W2
        self.b2 = self.b2 + delta_b2
        
        # --- Mutace pro W3 a b3 ---
        delta_W3 = (np_random.rand(*self.W3.shape) - 0.5) * mutation_rate
        delta_b3 = (np_random.rand(*self.b3.shape) - 0.5) * mutation_rate
        self.W3 = self.W3 + delta_W3
        self.b3 = self.b3 + delta_b3


        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))

        self.store()

    def store(self):
        # Všechny parametry sítě
        self.parameters = copy.deepcopy({
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "NAME": self.NAME,
            "N_HIDDEN_1": N_HIDDEN_1, 
            "N_HIDDEN_2": N_HIDDEN_2
        })

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        # Načtení všech parametrů
        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        self.W3 = np.array(self.parameters["W3"], dtype=float)
        self.b3 = np.array(self.parameters["b3"], dtype=float)
        self.NAME = str(self.parameters["NAME"])


    def calculate_score(self, distance, time, no):
        self.score = distance

    ##################### do těchto funkcí není potřeba zasahovat:
    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)