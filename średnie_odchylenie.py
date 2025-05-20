import math

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np







def odchylenie_standardowe(k_factor,a,b):
    # Wczytanie pliku Excela
    plik_excel = "dane.xlsx"  # Zmień na nazwę swojego pliku
    df = pd.read_excel(plik_excel)

    # Założenie, że kolumny w Excelu mają nazwy 'X' i 'Y'
    temp = np.array(df[['X', 'W']].values)
    t = temp[:, 0] # czas
    P = temp[:, 1] #  spadek ciśnienia



    V = k_factor *  np.sqrt(P)


    punkty = np.column_stack([t, V]).tolist()


    # Wyświetlenie wyników



    # Przykładowe punkty


    # Parametry prostej y = ax + b
    #a, b = np.polyfit(punkty[:, 0], punkty[:, 1], 1)

    # Współczynniki prostej Ax + By + C = 0 → a*x - y + b = 0


    A, B, C = a, -1, b

    # Obliczanie rzeczywistych odległości punktów od prostej
    def odległość_euklidesowa(x0, y0, A, B, C):
        return abs(A * x0 + B * y0 + C) / math.sqrt(A**2 + B**2)
    accu = []
    for (x,y) in punkty:
        z=odległość_euklidesowa(x, y, A, B, C) # dla każdego punktu obliczam odległość
        accu.append(z)  #lista z odległościami
    std_dev = np.std(accu)   # wyznaczenie standardowego odchylenia


    return float(std_dev)

###############################")


# a=5071.1 N
# b=-2324.3 N
a=5133.3
b=-2335.3

k = np.arange(30, 40, 0.1)  #  lista z wartościami k co 0.1 w przedziale


result = []
for i in k:
    result.append(odchylenie_standardowe(i, a, b))  # lista z odchyleniami standardowymi




print (result)

dictionary = dict(zip(k, result)) #łączę z k_fact z odchyleniem standardowym

min_key = min(dictionary, key=lambda x: dictionary[x]) # znajduję klucz dla minimalnej wartości
min_value = dictionary[min_key]  # znajduję minimalną wartość


print(f"Najmniejsza wartość: {min_value}, Klucz: {min_key}")





    #
    # # Obliczanie punktów rzutowanych prostopadle na prostą
    # rzutowane_punkty = []
    # for x0, y0 in punkty:
    #     x1 = (B * (B * x0 - A * y0) - A * C) / (A**2 + B**2)
    #     y1 = (A * (-B * x0 + A * y0) - B * C) / (A**2 + B**2)
    #     rzutowane_punkty.append([x1, y1])
    #
    # rzutowane_punkty = np.array(rzutowane_punkty)
    #
    # # Wizualizacja
    # plt.figure(figsize=(10, 5))
    # plt.scatter(punkty[:, 0], punkty[:, 1], label="Punkty", color="blue")
    # plt.plot(punkty[:, 0], a * punkty[:, 0] + b, label=f"y={a:.2f}x+{b:.2f}", color="red")
    #
    # for i in range(len(punkty)):
    #     plt.plot([punkty[i, 0], rzutowane_punkty[i, 0]], [punkty[i, 1], rzutowane_punkty[i, 1]],
    #              color="green", linestyle="dotted", label="Odległość" if i == 0 else "")
    #
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title(f"Odchylenie standardowe odległości: {std_dev:.3f}")
    # plt.legend()
    # plt.grid(True)
    #
    # # Wykres słupkowy odległości
    # plt.figure(figsize=(8, 4))
    # plt.bar(range(len(odległości)), odległości, color="purple", alpha=0.6)
    # plt.xlabel("Indeks punktu")
    # plt.ylabel("Odległość euklidesowa od prostej")
    # plt.title(f"Odchylenie standardowe odległości: {std_dev:.3f}")
    #
    # plt.show()
    # return std_dev

