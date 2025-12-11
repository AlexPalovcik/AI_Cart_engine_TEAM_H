from numpy import random as np_random
import random
import numpy as np
import copy
import string

# počet vstupů – ideálně = len(RAYCAST_ANGLES)
N_INPUTS = 9
N_ACTIONS = 4  # [up, down, left, right]
N_HIDDEN = 16  # Počet neuronů ve skryté vrstvě - zvoleno 16 pro dostatečnou komplexitu

# vždy pojmenováváme jako "AIbrain_jemnoteamu"
class AIbrain_shallowH:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # pro potreby náhdných znaků
        self.decider = 0
        self.x = 0
        self.y = 0
        self.speed = 0

        self.init_param()

    def init_param(self):
        """
        Inicializace vah a biasů pro 2 vrstvou síť (shallow NN).
        Používá se normální rozdělení pro inicializaci.
        """
        rng = np_random.default_rng()
        weight_scale = 1.0 / np.sqrt(N_INPUTS) # Scaling podle Kaiming/He pro ReLU

        # --- 1. vrstva (Vstup -> Skrytá) ---
        # Váhy W1: (N_HIDDEN, N_INPUTS)
        self.W1 = rng.normal(loc=0.0, scale=weight_scale,
                             size=(N_HIDDEN, N_INPUTS))
        # Biasy b1: (N_HIDDEN,)
        self.b1 = np.zeros(N_HIDDEN)

        # --- 2. vrstva (Skrytá -> Výstup) ---
        # Váhy W2: (N_ACTIONS, N_HIDDEN)
        weight_scale_2 = 1.0 / np.sqrt(N_HIDDEN)
        self.W2 = rng.normal(loc=0.0, scale=weight_scale_2,
                             size=(N_ACTIONS, N_HIDDEN))
        # Biasy b2: (N_ACTIONS,)
        self.b2 = np.zeros(N_ACTIONS)


        self.NAME = "teamH_shallow_nn"

        # vždy uložit!
        self.store()

    @staticmethod
    def _relu(z):
        """
        Aktivační funkce ReLU (Rectified Linear Unit).
        """
        return np.maximum(z, 0.0)

    def decide(self, data):
        """
        Propagace vpřed (forward pass) přes 2-vrstvou neuronovou síť.
        Z: W2 @ ReLU(W1 @ x + b1) + b2
        """
        self.decider += 1

        # Vstupní data
        x = np.asarray(data, dtype=float).ravel()
        
        # Ošetření velikosti vstupu (stejné jako u původního kódu)
        n_w = N_INPUTS
        if x.size < n_w:
            x = np.concatenate([x, np.zeros(n_w - x.size)])
        elif x.size > n_w:
            x = x[:n_w]

        # Vstupní vrstva (x) je (N_INPUTS,)
        
        # 1. vrstva: Vstup -> Skrytá
        # Lineární kombinace: W1 @ x + b1
        z1 = self.W1.dot(x) + self.b1
        # Aktivace: ReLU
        h1 = self._relu(z1)  # h1 má shape (N_HIDDEN,)

        # 2. vrstva: Skrytá -> Výstup
        # Lineární kombinace: W2 @ h1 + b2
        z2 = self.W2.dot(h1) + self.b2  # z2 má shape (N_ACTIONS,)

        # Vracíme raw výstupy; AI_car pak aplikuje threshold > 0.5
        return z2

    def mutate(self):
        """
        Mutace: všechny váhy a biasy (W1, b1, W2, b2) se malé náhodně posunou.
        """
        mutation_rate = 0.25 # Rozmezí perturbace [-0.125, 0.125]

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


        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=3))

        self.store()

    def store(self):
        # Všechny parametry sítě, co se mají ukládat do .npz
        self.parameters = copy.deepcopy({
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "NAME": self.NAME,
            "N_HIDDEN": N_HIDDEN # Uložení dimenze pro kontrolu
        })

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        # Zde nastavit co chceme ukládat:
        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        self.NAME = str(self.parameters["NAME"])
        
        # Volitelně kontrola, zda se neměnila topologie sítě
        # N_HIDDEN_LOADED = int(self.parameters.get("N_HIDDEN", N_HIDDEN))
        # if N_HIDDEN_LOADED != N_HIDDEN:
        #    print(f"Varování: Načtená síť má {N_HIDDEN_LOADED} skrytých neuronů, očekáváno {N_HIDDEN}.")


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